import os, io, gc, importlib.util
import cv2
import torch
import numpy as np
import requests
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import RobertaTokenizer
from huggingface_hub import hf_hub_download
from deep_translator import GoogleTranslator
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

PLANTNET_API_KEY = os.environ.get("PLANTNET_API_KEY", "").strip()
CACHE_DIR = "/tmp/models_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

scold_model_path = hf_hub_download(repo_id="enalis/scold", filename="model.py", local_dir=CACHE_DIR)
scold_weights_path = hf_hub_download(repo_id="enalis/scold", filename="scold.pth", local_dir=CACHE_DIR)

spec_s = importlib.util.spec_from_file_location("scold_mod", scold_model_path)
scold_mod = importlib.util.module_from_spec(spec_s)
spec_s.loader.exec_module(scold_mod)

SCOLD_MODEL = scold_mod.LVL()
state = torch.load(scold_weights_path, map_location="cpu", weights_only=True)
SCOLD_MODEL.load_state_dict(state, strict=True)
SCOLD_MODEL.eval()
del state
gc.collect()

vocab_file = hf_hub_download(repo_id="roberta-base", filename="vocab.json", local_dir=CACHE_DIR)
merges_file = hf_hub_download(repo_id="roberta-base", filename="merges.txt", local_dir=CACHE_DIR)
SCOLD_TOKENIZER = RobertaTokenizer(vocab_file=vocab_file, merges_file=merges_file)
SCOLD_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

ARABIC_PLANTS_DICT = {
    "solanum lycopersicum": ("طماطم", "Tomato"),
    "solanum tuberosum": ("بطاطس", "Potato"),
    "capsicum": ("فلفل (رومي / حار)", "Pepper"),
    "solanum melongena": ("باذنجان", "Eggplant"),
    "cucumis sativus": ("خيار", "Cucumber"),
    "cucurbita": ("كوسة / قرع", "Squash / Zucchini"),
    "citrullus": ("بطيخ", "Watermelon"),
    "cucumis melo": ("شمام / كنتالوب", "Cantaloupe / Melon"),
    "abelmoschus": ("بامية", "Okra"),
    "corchorus": ("ملوخية", "Molokhia"),
    "lactuca": ("خس", "Lettuce"),
    "spinacia": ("سبانخ", "Spinach"),
    "zea mays": ("ذرة شامية", "Corn / Maize"),
    "triticum": ("قمح", "Wheat"),
    "oryza": ("أرز", "Rice"),
    "gossypium": ("قطن", "Cotton"),
    "vitis": ("عنب", "Grape"),
    "fragaria": ("فراولة", "Strawberry"),
    "mangifera": ("مانجو", "Mango"),
    "citrus": ("موالح (برتقال / يوسفي / ليمون)", "Citrus"),
    "malus": ("تفاح", "Apple"),
    "musa": ("موز", "Banana"),
    "psidium": ("جوافة", "Guava"),
    "punica": ("رمان", "Pomegranate"),
    "ficus carica": ("تين برشومي", "Fig"),
    "olea": ("زيتون", "Olive"),
    "mentha": ("نعناع", "Mint"),
    "ocimum": ("ريحان", "Basil"),
    "rosa": ("ورد بلدي", "Rose")
}

DISEASES_CATALOG = [
    {"prompt": "a healthy green {plant} leaf with no spots, yellowing, or disease", "ar": "نبات سليم (لا توجد إصابة مرضية)", "en": "Healthy Leaf", "causes": "لا توجد أسباب مرضية؛ النبات في حالة صحية جيدة نتيجة توازن الري والتسميد والتهوية."},
    {"prompt": "a {plant} leaf infected with early blight showing dark brown concentric target-like rings and yellowing", "ar": "اللفحة المبكرة (Early Blight)", "en": "Early Blight (Alternaria)", "causes": "فطر (Alternaria) ينشط مع ارتفاع الرطوبة، درجات الحرارة الدافئة (24-29°م)، بقاء بقايا محاصيل مصابة في التربة، وري الأوراق مباشرة."},
    {"prompt": "a {plant} leaf infected with late blight showing large irregular dark water-soaked lesions and white fungal mold", "ar": "اللفحة المتأخرة (Late Blight)", "en": "Late Blight (Phytophthora)", "causes": "كائن شبه فطري (Phytophthora) ينتشر بسرعة في الجو البارد والرطب جداً (رطوبة فوق 90%) مع وجود ندى كثيف أو أمطار."},
    {"prompt": "a {plant} leaf with septoria leaf spot showing many small circular gray spots with dark brown borders", "ar": "تبقع الأوراق السبتوري (Septoria Leaf Spot)", "en": "Septoria Leaf Spot", "causes": "فطر (Septoria) ينتقل عبر تناثر قطرات المياه من التربة الملوثة إلى الأوراق السفلية مع الرطوبة العالية وازدحام النباتات."},
    {"prompt": "a {plant} leaf covered with powdery mildew white flour-like fungal patches", "ar": "البياض الدقيقي (Powdery Mildew)", "en": "Powdery Mildew", "causes": "فطريات البياض الدقيقي التي تنمو في الأجواء الدافئة والجافة نهاراً مع رطوبة عالية ليلاً، وقلة تعرض الأوراق للشمس والتهوية الجيدة."},
    {"prompt": "a {plant} leaf infected with downy mildew showing angular yellow spots on top and purplish-gray fuzz underneath", "ar": "البياض الزغبي (Downy Mildew)", "en": "Downy Mildew", "causes": "كائنات شبه فطرية تنشط في الجو الرطب المائل للبرودة، مع وجود بلل مستمر على الأوراق بسبب الري الرذاذي أو الضباب الكثيف."},
    {"prompt": "a {plant} leaf with bacterial leaf spot showing small dark water-soaked angular spots with yellow halos", "ar": "التبقع البكتيري (Bacterial Leaf Spot)", "en": "Bacterial Leaf Spot", "causes": "بكتيريا (Xanthomonas / Pseudomonas) تدخل عبر الجروح أو الثغور، وتنتقل عبر البذور الملوثة، أدوات التقليم غير المعقمة، أو رذاذ الري العلوي."},
    {"prompt": "a {plant} leaf damaged by spider mites or leaf miners showing tiny yellow stippling dots or white trails", "ar": "إصابة حشرية / أكاروسية (العنكبوت الأحمر أو صانعات الأنفاق)", "en": "Spider Mites / Leaf Miner Damage", "causes": "تغذية العنكبوت الأحمر أو يرقات صانعات الأنفاق على عصارة وأنسجة الورقة، وتنشط هذه الآفات في الجو الحار والجاف."},
    {"prompt": "a {plant} leaf infected with rust disease showing orange or reddish-brown powdery pustules", "ar": "مرض الصدأ (Leaf Rust)", "en": "Leaf Rust", "causes": "فطريات الصدأ التي تنتقل جراثيمها بالرياح وتنبت عند وجود رطوبة حرة (ندى) على سطح الورقة لعدة ساعات."},
    {"prompt": "a {plant} leaf infected with virus showing yellow mosaic mottling, curling, and leaf distortion", "ar": "إصابة فيروسية (تجعد أو تبرقش الأوراق الفيروسي)", "en": "Viral Mosaic / Leaf Curl Virus", "causes": "انتقال الفيروس عبر الحشرات الثاقبة الماصة (مثل الذبابة البيضاء وحشرة المن) أو استخدام شتلات وأدوات زراعية مصابة."},
    {"prompt": "a {plant} leaf showing nutrient deficiency or drought stress with yellowing between veins or dry scorched brown edges", "ar": "نقص عناصر غذائية أو إجهاد مائي / ملوحة (غير معدي)", "en": "Nutrient Deficiency / Abiotic Stress", "causes": "نقص عناصر أساسية في التربة (مثل النيتروجين، المغنيسيوم، الحديد، أو البوتاسيوم)، أو عدم انتظام الري وارتفاع ملوحة التربة."}
]

def isolate_leaf(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (14, 25, 25), (98, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > (arr.shape[0] * arr.shape[1] * 0.07):
            x, y, w, h = cv2.boundingRect(largest)
            px, py = int(w * 0.08), int(h * 0.08)
            return img.crop((max(0, x - px), max(0, y - py), min(arr.shape[1], x + w + px), min(arr.shape[0], y + h + py)))
    return img

def translate_plant(sci_name: str, common_names: list):
    sci_lower = sci_name.lower()
    for sci_key, (ar_name, en_name) in ARABIC_PLANTS_DICT.items():
        if sci_key in sci_lower:
            return ar_name, en_name
    en_candidate = common_names[0] if common_names else sci_name
    try:
        ar_tr = GoogleTranslator(source="en", target="ar").translate(en_candidate)
        if ar_tr:
            return ar_tr.strip(), en_candidate
    except Exception:
        pass
    return en_candidate, sci_name

def identify_plant(cropped_img: Image.Image, original_img: Image.Image):
    if PLANTNET_API_KEY:
        for test_img in [cropped_img, original_img]:
            for organ in ["leaf", "auto"]:
                try:
                    buf = io.BytesIO()
                    test_img.save(buf, format="JPEG", quality=92)
                    res = requests.post(
                        "https://my-api.plantnet.org/v2/identify/all",
                        params={"api-key": PLANTNET_API_KEY},
                        files=[("images", ("leaf.jpg", buf.getvalue(), "image/jpeg"))],
                        data=[("organs", organ)],
                        timeout=15
                    )
                    if res.status_code == 200:
                        results = res.json().get("results", [])
                        if results:
                            sp = results[0].get("species", {})
                            sci = sp.get("scientificNameWithoutAuthor", "")
                            if sci:
                                return translate_plant(sci, sp.get("commonNames", []))
                except Exception:
                    continue
    return "نبات غير محدد", "Unknown Plant"

@torch.inference_mode()
def diagnose_disease(cropped_img: Image.Image, original_img: Image.Image, plant_en: str):
    prompts = [d["prompt"].format(plant=plant_en) for d in DISEASES_CATALOG]
    tokens = SCOLD_TOKENIZER(prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    imgs_batch = torch.cat([SCOLD_TRANSFORM(cropped_img).unsqueeze(0), SCOLD_TRANSFORM(original_img).unsqueeze(0)], dim=0)
    img_f, txt_f = SCOLD_MODEL(imgs_batch, tokens["input_ids"], tokens["attention_mask"])
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    sims = (img_f @ txt_f.T).mean(dim=0).numpy()
    return DISEASES_CATALOG[int(np.argmax(sims))]

app = FastAPI(title="Plant Diagnosis API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health_check():
    return {"status": "online", "message": "Plant Diagnosis API is running 24/7"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    original_img = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
    cropped_img = isolate_leaf(original_img)
    plant_ar, plant_en = identify_plant(cropped_img, original_img)
    disease_obj = diagnose_disease(cropped_img, original_img, plant_en)
    return {
        "plant_ar": plant_ar,
        "plant_en": plant_en,
        "disease_ar": disease_obj["ar"],
        "disease_en": disease_obj["en"],
        "causes": disease_obj["causes"]
    }
