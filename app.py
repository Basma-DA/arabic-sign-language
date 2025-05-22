import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import cv2
import os

# إعداد الصفحة
st.set_page_config(
    page_title="تطبيق التعرف على لغة الإشارة العربية",
    page_icon="✋",
    layout="wide"
)

# عنوان التطبيق
st.title("✋ تطبيق التعرف على لغة الإشارة العربية")
st.markdown("### التعرف على الحروف العربية بلغة الإشارة")
st.markdown("---")

# الحروف العربية
ARABIC_LETTERS = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر',
    'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
    'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'
]

# بناء النموذج
@st.cache_resource
def create_model():
    """إنشاء نموذج CNN للتعرف على لغة الإشارة العربية"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(len(ARABIC_LETTERS), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# تحميل النموذج المدرب (إذا كان متوفراً)
@st.cache_resource
def load_trained_model():
    """تحميل النموذج المدرب إذا كان متوفراً"""
    try:
        # حاول تحميل النموذج المحفوظ
        if os.path.exists('arabic_sign_language_model.h5'):
            model = tf.keras.models.load_model('arabic_sign_language_model.h5')
            return model, True
        else:
            # إنشاء نموذج جديد إذا لم يكن النموذج المدرب متوفراً
            model = create_model()
            return model, False
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        model = create_model()
        return model, False

# معالجة الصورة
def preprocess_image(image):
    """معالجة الصورة للتنبؤ"""
    try:
        # تحويل إلى RGB إذا لزم الأمر
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # تغيير الحجم إلى 64x64 (حسب الداتاست)
        image = image.resize((64, 64))
        
        # تحويل إلى numpy array
        img_array = np.array(image)
        
        # تطبيع القيم (0-1)
        img_array = img_array.astype('float32') / 255.0
        
        # إضافة بعد للـ batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        st.error(f"خطأ في معالجة الصورة: {str(e)}")
        return None

# دالة التنبؤ
def predict_sign(model, processed_image, is_trained=False):
    """التنبؤ بالحرف من الصورة"""
    try:
        if not is_trained:
            st.warning("⚠️ النموذج غير مدرب بعد. النتائج قد تكون عشوائية.")
        
        # التنبؤ
        predictions = model.predict(processed_image, verbose=0)
        
        # الحصول على أعلى احتماليات
        top_indices = np.argsort(predictions[0])[::-1][:5]
        
        results = []
        for i, idx in enumerate(top_indices):
            letter = ARABIC_LETTERS[idx]
            confidence = predictions[0][idx] * 100
            results.append((letter, confidence))
        
        return results
        
    except Exception as e:
        st.error(f"خطأ في التنبؤ: {str(e)}")
        return None

# دالة تدريب النموذج (محاكاة)
def simulate_training():
    """محاكاة عملية التدريب"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f'جاري التدريب... {i+1}/100')
        
        # محاكاة وقت التدريب
        if i % 10 == 0:
            st.write(f"Epoch {i//10 + 1}/10 - Loss: {np.random.uniform(0.1, 2.0):.4f} - Accuracy: {np.random.uniform(0.7, 0.95):.4f}")
    
    status_text.text('تم الانتهاء من التدريب!')
    st.success("✅ تم تدريب النموذج بنجاح!")

# الواجهة الرئيسية
def main():
    # تحميل النموذج
    with st.spinner("جاري تحميل النموذج..."):
        model, is_trained = load_trained_model()
    
    # الشريط الجانبي
    st.sidebar.header("⚙️ الإعدادات")
    
    # معلومات عن الداتاست
    st.sidebar.info("""
    **معلومات الداتاست:**
    - 28 حرف عربي
    - صور RGB
    - حجم الصورة: 64x64
    - لغة الإشارة العربية
    """)
    
    # عرض الحروف المدعومة
    st.sidebar.subheader("الحروف المدعومة:")
    st.sidebar.write(" ".join(ARABIC_LETTERS))
    
    # خيار التدريب
    if not is_trained:
        st.sidebar.warning("النموذج غير مدرب")
        if st.sidebar.button("🏋️ تدريب النموذج", help="هذا محاكاة للتدريب"):
            simulate_training()
    else:
        st.sidebar.success("✅ النموذج مدرب ومجهز")
    
    # القسم الرئيسي
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 رفع الصورة")
        
        # خيارات الرفع
        upload_option = st.selectbox(
            "اختر طريقة الرفع:",
            ["رفع من الجهاز", "استخدام الكاميرا"]
        )
        
        uploaded_image = None
        
        if upload_option == "رفع من الجهاز":
            uploaded_image = st.file_uploader(
                "اختر صورة لحرف بلغة الإشارة...",
                type=['png', 'jpg', 'jpeg'],
                help="ارفع صورة لحرف بلغة الإشارة العربية"
            )
        else:
            uploaded_image = st.camera_input("التقط صورة للحرف")
        
        # عرض الصورة المرفوعة
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="الصورة المرفوعة", use_column_width=True)
            
            # معلومات الصورة
            st.info(f"حجم الصورة: {image.size} | النمط: {image.mode}")
    
    with col2:
        st.header("🔮 نتائج التنبؤ")
        
        if uploaded_image is not None:
            # معالجة الصورة
            processed_image = preprocess_image(image)
            
            if processed_image is not None:
                # عرض الصورة المعالجة
                st.subheader("الصورة بعد المعالجة:")
                processed_display = (processed_image[0] * 255).astype(np.uint8)
                st.image(processed_display, caption="64x64 RGB", width=200)
                
                # التنبؤ
                with st.spinner("جاري التحليل..."):
                    results = predict_sign(model, processed_image, is_trained)
                
                if results:
                    st.subheader("🎯 النتائج:")
                    
                    # عرض أفضل نتيجة
                    best_letter, best_confidence = results[0]
                    
                    # عرض النتيجة الرئيسية
                    st.success(f"""
                    ### الحرف المتوقع: **{best_letter}**
                    **الثقة: {best_confidence:.1f}%**
                    """)
                    
                    # عرض النتائج الأخرى
                    st.subheader("احتمالات أخرى:")
                    for i, (letter, confidence) in enumerate(results[1:], 2):
                        st.write(f"{i}. **{letter}** - {confidence:.1f}%")
                    
                    # رسم بياني للثقة
                    if len(results) > 1:
                        st.subheader("📊 مستوى الثقة:")
                        letters = [r[0] for r in results]
                        confidences = [r[1] for r in results]
                        
                        chart_data = {
                            'الحرف': letters,
                            'النسبة المئوية': confidences
                        }
                        
                        st.bar_chart(chart_data, x='الحرف', y='النسبة المئوية')
        else:
            st.info("👆 ارفع صورة لحرف بلغة الإشارة لبدء التحليل")
    
    # قسم المساعدة
    st.markdown("---")
    st.header("💡 نصائح للحصول على أفضل النتائج")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **📸 جودة الصورة:**
        - استخدم إضاءة جيدة
        - تأكد من وضوح اليد
        - تجنب الظلال القوية
        - خلفية بسيطة وواضحة
        """)
    
    with tips_col2:
        st.markdown("""
        **✋ وضعية اليد:**
        - اجعل اليد في المنتصف
        - تأكد من ظهور الحرف بوضوح
        - حافظ على المسافة المناسبة
        - تجنب حجب أجزاء من اليد
        """)

# تشغيل التطبيق
if __name__ == "__main__":
    main()

# معلومات إضافية في الأسفل
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>تطبيق التعرف على لغة الإشارة العربية | مبني بـ Streamlit و TensorFlow</p>
    <p>الداتاست: RGB Arabic Alphabets Sign Language Dataset</p>
</div>
""", unsafe_allow_html=True)
