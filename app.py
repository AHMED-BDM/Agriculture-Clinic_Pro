import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import datetime
import pickle

# ------------------------------------------------------------------------------
# إعداد الصفحة
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Pro Ag-Clinic AI", page_icon="🌿", layout="wide")

# ------------------------------------------------------------------------------
# اللغة – زر في الشريط العلوي بدلاً من الـ sidebar
# ------------------------------------------------------------------------------
# نضيف زر اختيار اللغة في العمود العلوي
col_lang1, col_lang2, col_lang3 = st.columns([1, 1, 8])
with col_lang2:
    lang = st.selectbox("", ["English", "العربية"], index=0, label_visibility="collapsed")

is_ar = lang == "العربية"

# قاموس النصوص (UI)
ui = {
    "title": "🌿 العيادة الزراعية الذكية الاحترافية" if is_ar else "🌿 Professional Agriculture AI Clinic",
    "subtitle": "نظام التحليل المرضي والاستشارة الفنية الدقيقة" if is_ar else "Detailed Pathological Analysis & Expert Consultation System",
    "input_header": "📋 إدخال بيانات الحقل" if is_ar else "📋 Input Field Data",
    "upload_label": "ارفع صورة الورقة المصابة (JPG/PNG)" if is_ar else "Upload Leaf Specimen (JPG/PNG)",
    "env_expander": "مستشعرات البيئة (إدخال يدوي)" if is_ar else "Environment Sensors (Manual Entry)",
    "temp": "درجة الحرارة المحيطة (مئوية)" if is_ar else "Ambient Temperature (°C)",
    "soil_label": "تكوين التربة" if is_ar else "Soil Composition",
    "soil_options": ["طينية", "رملية", "طميية", "غرينية"] if is_ar else ["Clay", "Sandy", "Loamy", "Silty"],
    "water_label": "مستوى الري/الرطوبة" if is_ar else "Irrigation/Moisture Level",
    "water_options": ["منخفض", "متوسط", "عالي", "مشبع بالمياه"] if is_ar else ["Low", "Medium", "High", "Waterlogged"],
    "report_header": "🔍 التقرير الفني للتحليل" if is_ar else "🔍 Technical Analysis Report",
    "btn_analyze": "بدء التحليل العميق" if is_ar else "EXECUTE DEEP ANALYSIS",
    "spinner": "جاري استدعاء قاعدة البيانات الزراعية..." if is_ar else "Accessing Agricultural Knowledge Base...",
    "wait": "في انتظار رفع صورة العينة للتشخيص..." if is_ar else "Awaiting leaf specimen for diagnosis...",
    "footer": "© 2026 النظم الزراعية الذكية | قسم الزراعة الدقيقة" if is_ar else "© 2026 Smart Agri-Systems | Expert Module | Precision Agriculture Division",
    "download_report": "تحميل التقرير (HTML)" if is_ar else "Download Report (HTML)",
    "print_report": "طباعة التقرير" if is_ar else "Print Report",
    # نصوص إضافية للتعليمات
    "how_to_title": "📖 كيفية الاستخدام" if is_ar else "📖 How to Use",
    "how_to_1": "1. اختر اللغة من القائمة العلوية." if is_ar else "1. Choose language from the top bar.",
    "how_to_2": "2. ارفع صورة واضحة لورقة النبات (JPG/PNG)." if is_ar else "2. Upload a clear leaf image (JPG/PNG).",
    "how_to_3": "3. أدخل بيانات البيئة (اختياري لكن يوصى به للحصول على توصيات مخصصة)." if is_ar else "3. Enter environmental data (optional but recommended for tailored advice).",
    "how_to_4": "4. اضغط على الزر 'بدء التحليل العميق' وانتظر التقرير." if is_ar else "4. Click 'EXECUTE DEEP ANALYSIS' and wait for the report.",
    "how_to_5": "5. يمكنك تحميل التقرير أو طباعته مباشرة." if is_ar else "5. You can download the report or print it directly.",
    "supported_title": "🌱 النباتات والأمراض المدعومة" if is_ar else "🌱 Supported Crops & Diseases",
    "supported_list": """
    - **الفلفل الحلو (Pepper bell)**: تبقع بكتيري، سليم.
    - **البطاطس (Potato)**: ندوة مبكرة، ندوة متأخرة، سليم.
    - **الطماطم (Tomato)**: تبقع بكتيري، ندوة مبكرة، ندوة متأخرة، عفن الأوراق، تبقع سبتوري، العنكبوت الأحمر، التبقع الهدفي، فيروس تجعد الأصفرار، فيروس الموزاييك، سليم.
    """ if is_ar else """
    - **Bell Pepper**: Bacterial spot, healthy.
    - **Potato**: Early blight, late blight, healthy.
    - **Tomato**: Bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, spider mites, target spot, tomato yellow leaf curl virus, tomato mosaic virus, healthy.
    """
}

# ------------------------------------------------------------------------------
# حالة الجلسة
# ------------------------------------------------------------------------------
if "saved_report" not in st.session_state:
    st.session_state.saved_report = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ------------------------------------------------------------------------------
# عتبات الثقة
# ------------------------------------------------------------------------------
PLANT_THRESHOLD = 0.7
DISEASE_THRESHOLD = 0.6

# ------------------------------------------------------------------------------
# تحميل النماذج وأسماء الكلاسات (مع التخزين المؤقت)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    plant_detector = tf.keras.models.load_model('plant_detector.keras')
    disease_model = tf.keras.models.load_model('Fplant_model.keras')
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return plant_detector, disease_model, class_names

plant_detector, disease_model, class_names = load_models()

# ------------------------------------------------------------------------------
# الترجمة العربية للأمراض
# ------------------------------------------------------------------------------
arabic_classes = {
    "Pepper__bell___Bacterial_spot": "فلفل حلو - تبقع بكتيري",
    "Pepper__bell___healthy": "فلفل حلو - سليم",
    "Potato___Early_blight": "بطاطس - ندوة مبكرة",
    "Potato___Late_blight": "بطاطس - ندوة متأخرة",
    "Potato___healthy": "بطاطس - سليم",
    "Tomato_Bacterial_spot": "طماطم - تبقع بكتيري",
    "Tomato_Early_blight": "طماطم - ندوة مبكرة",
    "Tomato_Late_blight": "طماطم - ندوة متأخرة",
    "Tomato_Leaf_Mold": "طماطم - عفن الأوراق",
    "Tomato_Septoria_leaf_spot": "طماطم - تبقع سبتوري",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "طماطم - العنكبوت الأحمر ذو البقعتين",
    "Tomato__Target_Spot": "طماطم - التبقع الهدفي",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "طماطم - فيروس تجعد واصفرار الأوراق",
    "Tomato__Tomato_mosaic_virus": "طماطم - فيروس الموزاييك",
    "Tomato_healthy": "طماطم - سليم"
}

# ------------------------------------------------------------------------------
# دالة التقرير المفصل (كاملة)
# ------------------------------------------------------------------------------
def get_detailed_report(disease, temp, soil, water, conf, is_ar):
    # حالة غير معروف
    if disease == "UNKNOWN":
        if is_ar:
            return f"""<div class="report-container">
<h2 style="text-align: center;">📋 تقرير العيادة الزراعية المعتمد</h2>
<p style="text-align: center; opacity:0.7;">المهندس الزراعي المعتمد: أحمد عبد الحافظ</p>
<hr>
<p><strong>التشخيص الأساسي:</strong> غير معروف</p>
<p><strong>دقة النموذج الإحصائي:</strong> {conf*100:.2f}%</p>
<hr>
<h3>⚠️ لم يتم التعرف على العينة</h3>
<p>لم يستطع النموذج تصنيف هذه الصورة ضمن الفئات المدربة. الأسباب المحتملة:</p>
<ul>
<li>الصورة لا تظهر ورقة نبات بشكل واضح.</li>
<li>النبات غير مدعوم (الفئات المدعومة: فلفل، بطاطس، طماطم).</li>
<li>الصورة غير واضحة أو تحتوي على ضوضاء.</li>
<li>الإصابة غير نمطية أو أن الورقة تالفة بشدة.</li>
</ul>
<p><strong>التوصيات:</strong></p>
<ul>
<li>تأكد من أن الصورة تركز على ورقة واحدة مع إضاءة جيدة.</li>
<li>حاول رفع صورة أخرى تظهر الأعراض بوضوح.</li>
<li>إذا استمرت المشكلة، يرجى التواصل مع فريق الدعم الفني.</li>
</ul>
<hr>
<p style="font-style: italic;"><strong>القرار الهندسي النهائي:</strong> الثقة منخفضة جدًا، لا يمكن تقديم توصيات علاجية. يُرجى التأكد من صحة العينة.</p>
</div>"""
        else:
            return f"""<div class="report-container">
<h2 style="text-align: center;">📋 CERTIFIED AGRICULTURE CLINIC REPORT</h2>
<p style="text-align: center; opacity:0.7;">Certified Agricultural Engineer: Ahmed Abd Al-Hafez</p>
<hr>
<p><strong>Primary Diagnosis:</strong> Unknown / Unrecognized</p>
<p><strong>Statistical Confidence:</strong> {conf*100:.2f}%</p>
<hr>
<h3>⚠️ Specimen Not Recognized</h3>
<p>The model could not classify this image among the trained categories. Possible reasons:</p>
<ul>
<li>The image does not clearly show a plant leaf.</li>
<li>The plant is not supported (supported crops: pepper, potato, tomato).</li>
<li>The image is blurry or contains noise.</li>
<li>The infection is atypical or the leaf is severely damaged.</li>
</ul>
<p><strong>Recommendations:</strong></p>
<ul>
<li>Ensure the image focuses on a single leaf with good lighting.</li>
<li>Try uploading another image that clearly shows the symptoms.</li>
<li>If the problem persists, please contact technical support.</li>
</ul>
<hr>
<p style="font-style: italic;"><strong>Final Engineering Verdict:</strong> Confidence is too low to provide treatment recommendations. Please verify the sample.</p>
</div>"""

    # التحضير للتقارير المعروفة
    disease_en = disease.replace("_", " ")
    disease_ar = arabic_classes.get(disease, disease_en)

    if is_ar:
        html = f"""<div class="report-container">
<h2 style="text-align: center;">📋 تقرير العيادة الزراعية المعتمد</h2>
<p style="text-align: center; opacity:0.7;">المهندس الزراعي المعتمد: أحمد عبد الحافظ</p>
<hr>
<p><strong>التشخيص الأساسي:</strong> {disease_ar}</p>
<p><strong>دقة النموذج الإحصائي:</strong> {conf*100:.2f}%</p>
<p><strong>الظروف الحقلية المسجلة:</strong> درجة الحرارة: {temp}°C | نوع التربة: {soil} | مستوى الري: {water}</p>
<hr>
"""

        # التوصيات حسب المرض
        if "healthy" in disease:
            html += """
<h3>✅ التقييم الفسيولوجي – سليم</h3>
<p>العينة المدروسة تظهر <strong>كفاءة تمثيل ضوئي ممتازة</strong> وغياب أي علامات إجهاد حيوي أو غير حيوي. الجهاز الوعائي يعمل بكفاءة عالية.</p>
<h4>📌 توصيات إستراتيجية للحفاظ على الصحة النباتية:</h4>
<ul>
<li><strong>التوازن الغذائي الدقيق:</strong> الحفاظ على نسب N-P-K متوازنة (1:1:1 خلال النمو الخضري، ثم 1:2:2 بعد التزهير). التركيز على إضافة الكالسيوم (نترات الكالسيوم 2-3 كجم/1000 م²) والبورون (حمض البوريك 0.5-1 جم/لتر) لضمان جودة الثمار.</li>
<li><strong>المنشطات الحيوية:</strong> تطبيق الأحماض الأمينية (1 لتر/فدان) وهيمات البوتاسيوم (2-3 كجم/فدان) لتحسين امتصاص الجذور وزيادة مقاومة الإجهادات البيئية.</li>
<li><strong>المراقبة الوقائية:</strong> الفحص الحقل المنتظم كل 48 ساعة لاكتشاف أي تجمعات مبكرة للآفات. استخدام المصائد اللاصقة الصفراء (15-20 مصيدة/فدان) لمراقبة الذبابة البيضاء والمن. تركيب شبكات حشرية (50 ميكرون) على مداخل الصوبات.</li>
<li><strong>إدارة الري:</strong> الري بالتنقيط للحفاظ على جفاف الأوراق. الري صباحاً للسماح بتبخر الرطوبة. في الأراضي الثقيلة، إدخال فترات جفاف بين الريات لتهوية الجذور وتجنب الإجهاد الناتج عن نقص الأكسجين.</li>
<li><strong>توصيات مستقبلية:</strong> تحليل التربة سنوياً لتعديل برامج التسميد. اعتماد أصناف مقاومة للأمراض السائدة في الموسم التالي.</li>
</ul>"""

        elif "Late_blight" in disease:
            html += """
<h3>🚨 إنذار مرضي – الندوة المتأخرة (Phytophthora infestans)</h3>
<p><strong>المسبب:</strong> فطر بيضي (Oomycete) يتطور بسرعة في الرطوبة الجوية >90% ودرجات حرارة معتدلة (15-25°م). يمكنه القضاء على الحقل بالكامل خلال 7-10 أيام في الظروف الملائمة.</p>
<p><strong>الأعراض التفصيلية:</strong> بقع مائية خضراء داكنة على الأوراق، تتحول إلى بنية سوداء مع هالة صفراء. على الساق تظهر بقع بنية مستطيلة. في الأجواء الرطبة، ينمو عشب أبيض (الجراثيم) على السطح السفلي للأوراق. الثمار تصاب ببقع بنية زيتية لامعة.</p>
<h4>🛠️ خطة المكافحة المتكاملة (الطوارئ):</h4>
<p><strong>1. التدخل الكيميائي العاجل (بالتناوب لتجنب المقاومة):</strong></p>
<ul>
<li><strong>ميتالاكسيل-إم + مانكوزيب</strong> (ريدوميل جولد) 250 جم/100 لتر – وقائي وعلاجي مبكر.</li>
<li><strong>سيموكسانيل + فاموكسادون</strong> (تاكوس) – تأثير علاجي قوي.</li>
<li><strong>بروباموكارب هيدروكلوريد</strong> (بروليفي) 150 مل/100 لتر – يحمي الجذور والساق.</li>
<li><strong>فوسيتيل-ألومنيوم</strong> (أليت) 200 جم/100 لتر – منشط مناعي.</li>
<li><strong>الرش كل 5-7 أيام، بالتناوب، مع مادة لاصقة (سيلكيت 0.5 مل/لتر).</strong></li>
</ul>
<p><strong>2. الإدارة الحقلية:</strong></p>
<ul>
<li><strong>الرطوبة:</strong> وقف الري بالرش فوراً، التحول إلى الري بالتنقيط، زيادة التهوية في الصوب.</li>
<li><strong>النظافة:</strong> قلع النباتات المصابة بشدة وحرقها خارج الحقل. تطهير الأدوات بالكحول 70% أو هيبوكلوريت الصوديوم.</li>
<li><strong>الدورة الزراعية:</strong> عدم زراعة محاصيل العائلة الباذنجانية (بطاطس، طماطم، فلفل) في نفس الحقل لمدة 3 سنوات على الأقل.</li>
</ul>
<p><strong>3. المكافحة الحيوية:</strong> رش مستحضرات تحتوي على <em>Trichoderma harzianum</em> أو <em>Bacillus subtilis</em> للحد من استمرار الفطر في التربة.</p>"""

        elif "Early_blight" in disease:
            html += """
<h3>🍂 الندوة المبكرة (Alternaria solani)</h3>
<p>مرض فطري شائع يظهر على الأوراق السفلية أولاً. البقع بنية داكنة ذات حلقات متحدة المركز تشبه هدف الرماية. ينتشر في درجات الحرارة الدافئة (25-30°م) والرطوبة المتوسطة.</p>
<h4>🛠️ خطة المكافحة:</h4>
<ul>
<li><strong>الإجراءات الثقافية:</strong> إزالة الأوراق السفلية المصابة، تحسين التهوية، الري بالتنقيط، تغطية التربة (المشش) لمنع تطاير الجراثيم.</li>
<li><strong>المكافحة الكيميائية:</strong> رش مبيدات تحتوي على (كلوروثالونيل، مانكوزيب، أو ديفينوكونازول) كل 7-10 أيام.</li>
<li><strong>التوصيات العضوية:</strong> يمكن استخدام مستخلص الثوم أو زيت النيم كبديل وقائي.</li>
</ul>"""

        elif "Bacterial_spot" in disease:
            html += """
<h3>🦠 التبقع البكتيري (Xanthomonas spp.)</h3>
<p>مرض بكتيري خطير يسبب بقع صغيرة مائية تتحول إلى بنية غامقة مع هالة صفراء. تنتقل البكتيريا بواسطة رذاذ الماء والأدوات الملوثة والحشرات.</p>
<h4>🛠️ خطة المكافحة:</h4>
<ul>
<li><strong>الري:</strong> تجنب الري العلوي تماماً، استخدم الري بالتنقيط.</li>
<li><strong>التطهير:</strong> تعقيم أدوات التقليم بمحلول 10% كلور، إزالة النباتات المصابة بشدة.</li>
<li><strong>المكافحة الكيميائية:</strong> رش مركبات النحاس (هيدروكسيد النحاس، أوكسي كلورور النحاس) بالتناوب مع المضادات الحيوية الحيوية (كاسوجاميسين) حيث يسمح بذلك.</li>
<li><strong>الوقاية:</strong> استخدام بذور وشتلات خالية من المرض، وتجنب الزراعة الكثيفة.</li>
</ul>"""

        elif "Spider_mites" in disease:
            html += """
<h3>🕷️ العنكبوت الأحمر ذو البقعتين (Tetranychus urticae)</h3>
<p>آفة دقيقة تمتص عصارة الأوراق مسببة بقع صفراء صغيرة ثم جفاف كامل. تفضل الظروف الحارة والجافة.</p>
<h4>🛠️ الإدارة المتكاملة:</h4>
<ul>
<li><strong>المكافحة الحيوية:</strong> إطلاق المفترس <em>Phytoseiulus persimilis</em> (10 أفراد/م²) عند أول إصابة.</li>
<li><strong>المكافحة الكيميائية:</strong> استخدام مبيدات أكاروسية متخصصة مثل (أبامكتين، سبيروميسيفين، هيكسيثياوكس) مع مراعاة التناوب.</li>
<li><strong>البيئة:</strong> رفع الرطوبة حول النبات بالرش بالماء (إن أمكن) لخلق ظروف غير مناسبة للعناكب.</li>
<li><strong>النظافة:</strong> إزالة الأعشاب الضارة لأنها عوائل بديلة.</li>
</ul>"""

        elif "Virus" in disease or "mosaic" in disease or "YellowLeaf" in disease:
            html += """
<h3>🚫 عدوى فيروسية جهازية</h3>
<p>الفيروسات (مثل فيروس تجعد الأصفرار، فيروس الموزاييك) تنتقل بواسطة الحشرات (الذبابة البيضاء، المن، التربس) أو الأدوات الملوثة. لا يوجد علاج كيميائي للنبات المصاب.</p>
<h4>🛠️ استراتيجية السيطرة (الوقاية خير من العلاج):</h4>
<ul>
<li><strong>مكافحة الناقل:</strong> استخدام مبيدات حشرية جهازية (إيميداكلوبريد، أسيتاميبريد) بالتناوب مع مبيدات تماس، مع وضع مصائد لاصقة صفراء.</li>
<li><strong>الاستئصال:</strong> قلع النباتات المصابة فوراً وحرقها خارج الحقل.</li>
<li><strong>الوقاية:</strong> استخدام شبكات حشرية (50 ميكرون) على الصوب، تعقيم الأدوات، شراء شتلات خالية من الفيروسات.</li>
</ul>"""

        elif "Leaf_Mold" in disease or "Septoria" in disease or "Target_Spot" in disease:
            html += """
<h3>🍄 أمراض الرطوبة والتبقعات الورقية</h3>
<p>تنتج عن فطريات تزدهر في الرطوبة العالية وسوء التهوية. تظهر بقع بنية أو صفراء مع هالات.</p>
<h4>🛠️ التوصيات:</h4>
<ul>
<li><strong>التهوية:</strong> تقليم الأوراق الكثيفة، فتح نوافذ الصوب، خفض كثافة الزراعة.</li>
<li><strong>الري:</strong> الري صباحاً وتجنب بلل الأوراق.</li>
<li><strong>المكافحة الكيميائية:</strong> رش مبيدات فطرية جهازية ووقائية (أزوكسيستروبين، تيبوكونازول، كلوروثالونيل).</li>
<li><strong>الزراعة:</strong> استخدام أصناف مقاومة إن وجدت.</li>
</ul>"""

        else:
            html += f"""
<h3>🍄 التشخيص: {disease_ar}</h3>
<p>بقع نخرية على الأوراق ناتجة عن مسببات فطرية أو بكتيرية. يعتمد العلاج على المسبب الدقيق ولكن يمكن اتباع البروتوكول العام التالي.</p>
<h4>🛠️ توصيات الخبراء للسيطرة:</h4>
<ul>
<li><strong>المكافحة الكيميائية (عند ظهور الأعراض):</strong> كلوروثالونيل، أزوكسيستروبين، أوكسي كلورور النحاس، مانكوزيب.</li>
<li><strong>الإجراءات الثقافية:</strong> تقليم الأوراق المصابة، تجنب الري بالرش، التغطية بالبلاستيك، تناوب المحاصيل.</li>
<li><strong>المكافحة الحيوية:</strong> رش مستحضرات <em>Bacillus subtilis</em> أو <em>Trichoderma</em>.</li>
</ul>"""

        # التوصيات البيئية
        html += "<h4>🌍 عوامل الذكاء البيئي وتوصيات مكيفة:</h4>"
        if temp > 38:
            html += f"<p>⚠️ <strong>إجهاد حراري ({temp}°م):</strong> استخدام <strong>سليكات البوتاسيوم</strong> (2 مل/لتر) لتقوية جدر الخلايا وتقليل النتح. رش الأحماض الأمينية والجبرلين لتخفيف الإجهاد. زيادة الري ليلاً لتبريد منطقة الجذور.</p>"
        elif temp < 10:
            html += f"<p>❄️ <strong>إجهاد برودة ({temp}°م):</strong> استخدام أغطية بلاستيكية ليلاً. رش سترات الكالسيوم لتعزيز صلابة النبات. تقليل الري لتجنب تعفن الجذور.</p>"
        else:
            html += f"<p>✅ درجة الحرارة الحالية ({temp}°م) ضمن النطاق المناسب للنمو. استمر في المراقبة.</p>"

        if soil == "طينية" and water in ["عالي", "مشبع بالمياه"]:
            html += "<p>⚠️ <strong>ميكانيكا التربة:</strong> خطر مرتفع لحدوث <strong>نقص الأكسجين (Hypoxia)</strong> وعفن الجذور في التربة الثقيلة. إضافة مادة عضوية (كمبوست) لتحسين الصرف. إطالة فترات الجفاف بين الريات. استخدام مراوح تهوية في الصوب لزيادة الأكسجين حول الجذور.</p>"
        elif soil == "رملية" and water == "منخفض":
            html += "<p>💧 <strong>إجهاد جفاف:</strong> التربة الرملية تستنزف الماء بسرعة. النبات يقترب من نقطة الذبول الدائم. زيادة وتيرة الري مع تقليل الكمية (الري المتكرر الخفيف). إضافة مواد حافظة للرطوبة مثل البوليمرات الماصة أو الكمبوست لتحسين احتفاظ التربة بالماء.</p>"
        elif soil == "طينية" and water == "منخفض":
            html += "<p>⚠️ <strong>إجهاد جفاف في تربة ثقيلة:</strong> التربة الطينية قد تتشقق وتجف بسرعة رغم قدرتها على الاحتفاظ بالماء. التغطية العضوية (القش) حول النباتات للحفاظ على رطوبة التربة ومنع التبخر.</p>"
        else:
            html += "<p>✅ ظروف التربة والري مناسبة حالياً. استمر في البرنامج المتبع.</p>"

        html += f"""<hr>
<p style="font-style: italic;"><strong>القرار الهندسي النهائي:</strong> مستوى الثقة {conf*100:.1f}%. التوصيات تستند إلى البروتوكولات الزراعية الدولية وممارسات الخبراء المعتمدة.</p>
<p style="font-size: 0.9em;"><strong>ملاحظة مهمة:</strong> هذه التوصيات استشارية. يُرجى الرجوع إلى مهندس زراعي معتمد لتحديد الجرعات وفق ظروفك الحقلية والتشريعات المحلية.</p>
</div>"""
        return html

    else:  # English version
        html = f"""<div class="report-container">
<h2 style="text-align: center;">📋 CERTIFIED AGRICULTURE CLINIC REPORT</h2>
<p style="text-align: center; opacity:0.7;">Certified Agricultural Engineer: Ahmed Abd Al-Hafez</p>
<hr>
<p><strong>Primary Diagnosis:</strong> {disease_en}</p>
<p><strong>Statistical Confidence:</strong> {conf*100:.2f}%</p>
<p><strong>Field Conditions:</strong> Temperature: {temp}°C | Soil: {soil} | Irrigation: {water}</p>
<hr>
"""

        if "healthy" in disease:
            html += """
<h3>✅ Physiological Assessment – Healthy</h3>
<p>The analyzed sample exhibits <strong>excellent photosynthetic efficiency</strong> and no signs of biotic or abiotic stress. The vascular system is fully functional.</p>
<h4>📌 Strategic Recommendations for Plant Health Maintenance:</h4>
<ul>
<li><strong>Precise Nutritional Balance:</strong> Maintain balanced N-P-K ratios (1:1:1 vegetative, 1:2:2 after flowering). Apply Calcium (calcium nitrate 2-3 kg/1000 m²) and Boron (0.5-1 g/L) to prevent blossom end rot.</li>
<li><strong>Biostimulants:</strong> Use amino acids (1 L/ha) and potassium humate (2-3 kg/ha) to enhance root absorption and stress tolerance.</li>
<li><strong>Preventive Monitoring:</strong> Scout every 48 hours. Use yellow sticky traps (15-20/acre) for whiteflies and aphids. Install insect-proof nets (50 mesh).</li>
<li><strong>Irrigation Management:</strong> Drip irrigation to keep foliage dry. Water in the morning. On heavy soils, allow dry periods between irrigations to avoid hypoxia.</li>
<li><strong>Future Steps:</strong> Annual soil analysis. Use resistant cultivars.</li>
</ul>"""

        elif "Late_blight" in disease:
            html += """
<h3>🚨 Pathogen Alert – Late Blight (Phytophthora infestans)</h3>
<p><strong>Etiology:</strong> An oomycete that thrives in high humidity (>90%) and moderate temperatures (15-25°C). Can destroy a field in 7-10 days. Spreads via wind‑ and water‑borne spores.</p>
<p><strong>Detailed Symptoms:</strong> Dark, water‑soaked lesions on leaves turning brown‑black with yellow halo. Brown elongated lesions on stems. In humid conditions, white sporangial growth on leaf undersides. Fruits develop oily brown spots leading to rot.</p>
<h4>🛠️ Integrated Management Plan (Emergency):</h4>
<p><strong>1. Immediate Chemical Intervention (rotate products):</strong></p>
<ul>
<li><strong>Metalaxyl-M + Mancozeb</strong> (Ridomil Gold) 250 g/100 L – preventive and early curative.</li>
<li><strong>Cymoxanil + Famoxadone</strong> (Tanos) – strong curative action.</li>
<li><strong>Propamocarb Hydrochloride</strong> (Previcur) 150 ml/100 L – protects roots and stems.</li>
<li><strong>Fosetyl-Aluminum</strong> (Aliette) 200 g/100 L – resistance inducer.</li>
<li><strong>Apply every 5-7 days, alternating, with a sticker (Silwet 0.5 ml/L).</strong></li>
</ul>
<p><strong>2. Field Management:</strong></p>
<ul>
<li><strong>Humidity:</strong> Stop overhead irrigation immediately; switch to drip. Increase greenhouse ventilation.</li>
<li><strong>Sanitation:</strong> Remove and burn severely infected plants. Disinfect tools with 70% alcohol or bleach.</li>
<li><strong>Crop Rotation:</strong> Avoid planting Solanaceae crops in the same field for at least 3 years.</li>
</ul>
<p><strong>3. Biological Control:</strong> Apply <em>Trichoderma harzianum</em> or <em>Bacillus subtilis</em> to suppress soil‑borne inoculum.</p>"""

        elif "Early_blight" in disease:
            html += """
<h3>🍂 Early Blight (Alternaria solani)</h3>
<p>A common fungal disease affecting lower leaves first. Brown spots with concentric rings (target‑like). Favors warm temperatures (25-30°C) and moderate humidity.</p>
<h4>🛠️ Management:</h4>
<ul>
<li><strong>Cultural:</strong> Remove infected lower leaves, improve aeration, drip irrigation, mulching.</li>
<li><strong>Chemical:</strong> Apply fungicides containing chlorothalonil, mancozeb, or difenoconazole every 7-10 days.</li>
<li><strong>Organic alternatives:</strong> Garlic extract or neem oil as preventive.</li>
</ul>"""

        elif "Bacterial_spot" in disease:
            html += """
<h3>🦠 Bacterial Spot (Xanthomonas spp.)</h3>
<p>A serious bacterial disease causing small water‑soaked lesions turning dark with yellow halo. Spread by water splash, contaminated tools, and insects.</p>
<h4>🛠️ Management:</h4>
<ul>
<li><strong>Irrigation:</strong> Avoid overhead watering; use drip irrigation.</li>
<li><strong>Sanitation:</strong> Disinfect pruning tools (10% bleach), remove severely infected plants.</li>
<li><strong>Chemical:</strong> Spray copper‑based products (copper hydroxide, copper oxychloride) alternating with kasugamycin where permitted.</li>
<li><strong>Prevention:</strong> Use disease‑free seeds/seedlings, avoid dense planting.</li>
</ul>"""

        elif "Spider_mites" in disease:
            html += """
<h3>🕷️ Two‑Spotted Spider Mite (Tetranychus urticae)</h3>
<p>Microscopic pest that sucks cell contents causing stippling, then leaf desiccation. Prefers hot, dry conditions.</p>
<h4>Integrated Pest Management:</h4>
<ul>
<li><strong>Biological control:</strong> Release predatory mite <em>Phytoseiulus persimilis</em> (10 individuals/m²) at first sign.</li>
<li><strong>Chemical control:</strong> Use specific acaricides (abamectin, spiromesifen, hexythiazox) with rotation.</li>
<li><strong>Environmental:</strong> Increase humidity by misting (if possible) to discourage mites.</li>
<li><strong>Weed removal:</strong> Eliminate alternative hosts.</li>
</ul>"""

        elif "Virus" in disease or "mosaic" in disease or "YellowLeaf" in disease:
            html += """
<h3>🚫 Systemic Viral Infection</h3>
<p>Viruses (e.g., Tomato Yellow Leaf Curl Virus, Mosaic virus) transmitted by insects (whiteflies, aphids, thrips) or contaminated tools. No cure for infected plants.</p>
<h4>🛠️ Control Strategy (Prevention is key):</h4>
<ul>
<li><strong>Vector control:</strong> Use systemic insecticides (imidacloprid, acetamiprid) alternating with contact products, plus yellow sticky traps.</li>
<li><strong>Eradication:</strong> Immediately remove and burn infected plants.</li>
<li><strong>Prevention:</strong> Install insect nets (50 mesh), disinfect tools, buy virus‑free seedlings.</li>
</ul>"""

        elif "Leaf_Mold" in disease or "Septoria" in disease or "Target_Spot" in disease:
            html += """
<h3>🍄 Humidity‑related Leaf Spots</h3>
<p>Fungal diseases that thrive in high humidity and poor air circulation. Brown or yellow spots with halos.</p>
<h4>🛠️ Recommendations:</h4>
<ul>
<li><strong>Ventilation:</strong> Prune dense foliage, open greenhouse vents, reduce planting density.</li>
<li><strong>Irrigation:</strong> Water in the morning, avoid wetting leaves.</li>
<li><strong>Chemical:</strong> Apply systemic and protective fungicides (azoxystrobin, tebuconazole, chlorothalonil).</li>
<li><strong>Resistance:</strong> Use resistant varieties if available.</li>
</ul>"""

        else:
            html += f"""
<h3>🍄 Diagnosis: {disease_en}</h3>
<p>Necrotic leaf lesions likely caused by fungal or bacterial pathogens. A general control protocol is provided below.</p>
<h4>🛠️ Expert Recommendations:</h4>
<ul>
<li><strong>Chemical:</strong> Chlorothalonil, azoxystrobin, copper oxychloride, mancozeb.</li>
<li><strong>Cultural:</strong> Prune infected leaves, avoid overhead irrigation, mulch, rotate crops.</li>
<li><strong>Biological:</strong> Apply <em>Bacillus subtilis</em> or <em>Trichoderma</em>.</li>
</ul>"""

        # Environmental recommendations
        html += "<h4>🌍 Environmental Intelligence Factors & Tailored Recommendations:</h4>"
        if temp > 38:
            html += f"<p>⚠️ <strong>Heat Stress ({temp}°C):</strong> Apply <strong>Potassium Silicate</strong> (2 ml/L) to strengthen cell walls. Spray amino acids and gibberellins. Increase night irrigation to cool roots.</p>"
        elif temp < 10:
            html += f"<p>❄️ <strong>Cold Stress ({temp}°C):</strong> Use plastic covers at night. Spray calcium citrate. Reduce irrigation.</p>"
        else:
            html += f"<p>✅ Current temperature ({temp}°C) is within optimal range. Continue monitoring.</p>"

        if soil == "Clay" and water in ["High", "Waterlogged"]:
            html += "<p>⚠️ <strong>Soil Mechanics:</strong> High risk of <strong>Hypoxia</strong> and root rot. Add organic matter (compost) to improve drainage. Increase dry periods between irrigations. Use ventilation fans in greenhouses.</p>"
        elif soil == "Sandy" and water == "Low":
            html += "<p>💧 <strong>Drought Stress:</strong> Sandy soils drain rapidly. Increase irrigation frequency with smaller amounts (light, frequent watering). Add moisture‑retaining materials like hydrogels or compost.</p>"
        elif soil == "Clay" and water == "Low":
            html += "<p>⚠️ <strong>Drought Stress in Heavy Soil:</strong> Clay soils may crack and dry quickly despite high water‑holding capacity. Apply organic mulch (straw) around plants to conserve moisture.</p>"
        else:
            html += "<p>✅ Soil and irrigation conditions are currently suitable. Continue your regular program.</p>"

        html += f"""<hr>
<p style="font-style: italic;"><strong>Final Engineering Verdict:</strong> Confidence level is {conf*100:.1f}%. Recommendations are based on international agronomic protocols and expert practices.</p>
<p style="font-size: 0.9em;"><strong>Important Note:</strong> These recommendations are advisory. Always consult a certified agricultural engineer for precise dosages based on your field conditions and local regulations.</p>
</div>"""
        return html

# ------------------------------------------------------------------------------
# CSS محسن – خطوط واضحة، إزالة الهوامش الزائدة
# ------------------------------------------------------------------------------
direction = "rtl" if is_ar else "ltr"
text_align = "right" if is_ar else "left"
font_family = "'Tajawal', 'Segoe UI', Tahoma, sans-serif" if is_ar else "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

img_base64 = get_base64_image("background.jpg")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&family=Segoe+UI:wght@400;500;700&display=swap');
    
    /* Reset margins */
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    .stApp {{
        background-image: url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    .stApp > div:first-child {{
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(2px);
    }}
    
    /* Hide default header and toolbar */
    .stApp header,
    .stApp .st-emotion-cache-1r6slb0,
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {{
        background: transparent !important;
        display: none !important;
    }}
    
    /* Columns */
    .row-widget.stHorizontal {{
        align-items: flex-start !important;
    }}
    
    [data-testid="stColumn"] {{
        background: rgba(255, 255, 255, 0.88);
        border-radius: 24px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: auto !important;
        align-self: flex-start;
    }}
    
    [data-testid="stColumn"]:hover {{
        box-shadow: 0 12px 28px rgba(0,0,0,0.1);
    }}
    
    /* Scrollable report area */
    .report-scrollable {{
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 10px;
        margin-top: 1rem;
        scrollbar-width: thin;
    }}
    
    .report-scrollable::-webkit-scrollbar {{
        width: 6px;
    }}
    
    .report-scrollable::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 3px;
    }}
    
    .report-scrollable::-webkit-scrollbar-thumb {{
        background: #2e7d32;
        border-radius: 3px;
    }}
    
    /* Base text – high contrast */
    html, body, [class*="css"], .stMarkdown, .stSubheader, .stTitle, .stCaption,
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError,
    .stSelectbox, .stSlider, .stTextInput, label,
    .stSidebar * {{
        color: #000000 !important;
        font-family: {font_family};
        direction: {direction};
        text-align: {text_align};
        font-size: 16px !important;
        line-height: 1.5 !important;
    }}
    
    h1 {{ font-size: 2.5rem !important; font-weight: 800 !important; }}
    h2 {{ font-size: 2rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 1.5rem !important; font-weight: 700 !important; }}
    h4 {{ font-size: 1.2rem !important; font-weight: 600 !important; }}
    
    /* Report container */
    .report-container {{
        background: #ffffff !important;
        border-radius: 28px;
        padding: 2rem;
        border: none;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.15);
        line-height: 1.7;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeSlideUp 0.5s cubic-bezier(0.2, 0.9, 0.4, 1.1);
    }}
    
    @keyframes fadeSlideUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .report-container:hover {{
        box-shadow: 0 25px 40px -12px rgba(0,0,0,0.2);
    }}
    
    .report-container hr {{
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #1b5e20, #2e7d32, #1b5e20);
        margin: 1.2rem 0;
    }}
    
    /* Buttons */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 40px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
    }}
    
    /* Images */
    img {{
        border-radius: 20px;
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        max-width: 100%;
    }}
    
    img:hover {{
        transform: scale(1.02);
    }}
    
    /* Info, warning, success boxes */
    .stInfo {{
        background: rgba(0,0,0,0.05) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        border-left: 6px solid #1b5e20 !important;
    }}
    
    .stWarning {{
        background: rgba(255, 243, 205, 0.9) !important;
        border-left: 6px solid #f39c12 !important;
    }}
    
    .stSuccess {{
        background: rgba(212, 237, 218, 0.9) !important;
        border-left: 6px solid #28a745 !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        font-weight: 600 !important;
        background: rgba(0,0,0,0.02) !important;
        border-radius: 12px !important;
    }}
    
    /* Print styles */
    @media print {{
        body * {{
            visibility: hidden;
        }}
        .report-container, .report-container * {{
            visibility: visible;
        }}
        .report-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            margin: 0;
            padding: 20px;
            box-shadow: none;
        }}
        .stApp, .stApp > div, [data-testid="stColumn"] {{
            background: white !important;
        }}
        .stButton, .stSidebar, .stMarkdown:has(button) {{
            display: none !important;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# تخطيط الصفحة الرئيسية
# ------------------------------------------------------------------------------
st.title(ui["title"])
st.markdown(ui["subtitle"])
st.markdown("---")

# عمودان: الأيسر للإدخال والتعليمات، الأيمن للتقرير
c1, c2 = st.columns([1, 1.4], gap="large")

with c1:
    st.subheader(ui["input_header"])
    uploaded_file = st.file_uploader(ui["upload_label"], type=["jpg","jpeg","png"])
    
    with st.expander(ui["env_expander"]):
        t_input = st.slider(ui["temp"], 0, 55, 26)
        s_input_raw = st.selectbox(ui["soil_label"], ui["soil_options"])
        w_input_raw = st.selectbox(ui["water_label"], ui["water_options"])
    
    # تعليمات الاستخدام والنباتات المدعومة
    st.markdown("---")
    st.markdown(f"### {ui['how_to_title']}")
    st.markdown(ui['how_to_1'])
    st.markdown(ui['how_to_2'])
    st.markdown(ui['how_to_3'])
    st.markdown(ui['how_to_4'])
    st.markdown(ui['how_to_5'])
    
    st.markdown(f"### {ui['supported_title']}")
    st.markdown(ui['supported_list'])

with c2:
    st.subheader(ui["report_header"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width=400, caption=f"ID: {uploaded_file.name}")
        
        if st.button(ui["btn_analyze"]):
            with st.spinner(ui["spinner"]):
                # Preprocess
                proc_img = img.convert("RGB").resize((224, 224))
                img_array = np.array(proc_img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Plant detection
                plant_pred = plant_detector.predict(img_array, verbose=0)
                plant_conf = float(plant_pred[0][0])
                is_plant = plant_conf > PLANT_THRESHOLD
                
                if not is_plant:
                    label = "UNKNOWN"
                    best_conf = plant_conf
                    if is_ar:
                        st.warning("⚠️ الصورة لا تبدو وكأنها ورقة نبات. يرجى رفع صورة واضحة لأحد المحاصيل المدعومة (فلفل، بطاطس، طماطم).")
                    else:
                        st.warning("⚠️ The image does not appear to be a plant leaf. Please upload a clear image of a supported crop (pepper, potato, tomato).")
                else:
                    # Disease classification
                    raw_preds = disease_model.predict(img_array, verbose=0)
                    best_idx = np.argmax(raw_preds)
                    best_conf = float(np.max(raw_preds))
                    label = class_names[best_idx]
                    
                    if best_conf < DISEASE_THRESHOLD:
                        label = "UNKNOWN"
                        if is_ar:
                            st.warning("⚠️ لم يتم التعرف على المرض بدقة كافية. قد تكون الصورة غير واضحة أو تحتوي على إصابة غير نمطية.")
                        else:
                            st.warning("⚠️ The disease could not be identified with enough confidence. The image may be unclear or show an atypical infection.")
                    else:
                        identified_text = arabic_classes.get(label, label) if is_ar else label.replace('___', ' | ')
                        st.success(f"{identified_text} ✓")
                
                # Generate report
                full_report = get_detailed_report(label, t_input, s_input_raw, w_input_raw, best_conf, is_ar)
                st.session_state.saved_report = full_report
                st.session_state.analysis_done = True
        
        # عرض التقرير
        if st.session_state.analysis_done and st.session_state.saved_report:
            st.markdown("---")
            st.markdown("**📊 نتائج التحليل**" if is_ar else "**📊 Analysis Results**")
            st.markdown(f'<div class="report-scrollable">{st.session_state.saved_report}</div>', unsafe_allow_html=True)
            
            # أزرار التصدير
            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agri_report_{timestamp}.html"
            with col_btn1:
                st.download_button(
                    label=ui["download_report"],
                    data=st.session_state.saved_report,
                    file_name=filename,
                    mime="text/html",
                    key="download_html"
                )
            with col_btn2:
                escaped_report = st.session_state.saved_report.replace('`', '\\`').replace('${', '\\${')
                print_js = f"""
                <script>
                function printReport() {{
                    var reportHTML = `{escaped_report}`;
                    var printWindow = window.open('', '_blank');
                    printWindow.document.write('<html><head><title>Agriculture Report</title>');
                    printWindow.document.write('<style>body {{ font-family: sans-serif; padding: 2rem; }} .report-container {{ max-width: 1000px; margin: auto; }}</style>');
                    printWindow.document.write('</head><body>');
                    printWindow.document.write(reportHTML);
                    printWindow.document.write('</body></html>');
                    printWindow.document.close();
                    printWindow.print();
                }}
                </script>
                <button onclick="printReport()" style="width:100%; background: linear-gradient(135deg, #1b5e20, #2e7d32); color: white; font-size: 1.1rem; font-weight: 600; padding: 0.75rem 1.5rem; border-radius: 40px; border: none; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.1); transition: all 0.3s ease;">
                    {ui["print_report"]}
                </button>
                """
                st.markdown(print_js, unsafe_allow_html=True)
    else:
        st.info(ui["wait"])

st.markdown("---")
st.caption(ui["footer"])
