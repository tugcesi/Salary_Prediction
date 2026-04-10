import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import plotly.express as px

# PAGE CONFIGURATION
st.set_page_config(
    page_title="💼 Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .result-box {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .result-salary {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .result-label {
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    .sidebar-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #3498db;
        margin-bottom: 15px;
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        width: 100%;
        height: 50px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Model'i yükle (pkl veya joblib)"""
    try:
        # Önce pkl'den yükle (features ile)
        with open('salary_pred.pkl', 'rb') as f:
            bundle = pickle.load(f)
        
        model = bundle['model']
        features = bundle['features']
        
        return model, features
        
    except FileNotFoundError:
        try:
            # Eğer pkl yoksa joblib'den yükle
            model = joblib.load('salary_pred.joblib')
            
            # Features'ları modelden al
            if hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
            else:
                features = None
            
            return model, features
            
        except FileNotFoundError:
            st.error("❌ salary_pred.pkl veya salary_pred.joblib bulunamadı!")
            return None, None
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        return None, None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_salary_range_interpretation(predicted_salary):
    """Maaş aralığı yorumlama"""
    if predicted_salary < 30000:
        return "💰 Giriş Seviyesi", "Genç profesyoneller"
    elif predicted_salary < 60000:
        return "💵 Orta Seviye", "Deneyimli çalışanlar"
    elif predicted_salary < 100000:
        return "💳 Üst Seviye", "Yönetici pozisyonları"
    else:
        return "👑 Executive", "Üst yönetim"

def get_job_level_emoji(level):
    """Seviye emojisi"""
    emojis = {
        'Junior': '🌱',
        'Mid': '📈',
        'Senior': '⭐',
        'Lead': '👨‍💼',
        'Manager': '👔',
        'Director': '🎯',
        'Executive': '👑'
    }
    return emojis.get(level, '💼')

def predict_salary(model, model_features, input_dict):
    """Maaş tahmini yap"""
    try:
        # Eğer features bilgisi varsa
        if model_features:
            feature_vector = []
            for feature in model_features:
                if feature in input_dict:
                    feature_vector.append(input_dict[feature])
                else:
                    feature_vector.append(0)
            X = np.array(feature_vector).reshape(1, -1)
        else:
            # Features bilgisi yoksa, input_dict'ten doğrudan array oluştur
            X = np.array(list(input_dict.values())).reshape(1, -1)
        
        # Tahmin yap
        prediction = model.predict(X)
        
        # Array'den scalar çıkar
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()
        else:
            prediction = float(prediction)
        
        prediction = max(prediction, 0)
        
        return prediction
        
    except Exception as e:
        st.error(f"❌ Tahmin hatası: {str(e)}")
        return None

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>💼 Salary Predictor</h1>
        <p>🔮 Kişisel ve profesyonel özelliklerinize göre maaş tahmini yapın</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model yükle
    model, model_features = load_model()
    if model is None:
        st.stop()
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">📊 Kişisel ve Profesyonel Bilgiler</div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        input_dict = {}
        
        # ============================================================
        # KİŞİSEL BİLGİLER
        # ============================================================
        st.markdown("**👤 Kişisel Bilgiler**")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("📅 Yaş", 18, 70, 30)
            input_dict['age'] = age
        
        with col2:
            years_experience = st.slider("⏱️ Tecrübe (Yıl)", 0, 50, 5)
            input_dict['years_experience'] = years_experience
        
        col1, col2 = st.columns(2)
        with col1:
            education_level = st.selectbox("🎓 Eğitim Seviyesi",
                                          ['High School', 'Bachelor', 'Master', 'PhD'],
                                          help="En yüksek eğitim seviyesi")
            education_mapping = {
                'High School': 1,
                'Bachelor': 2,
                'Master': 3,
                'PhD': 4
            }
            input_dict['education_level'] = education_mapping.get(education_level, 1)
        
        with col2:
            gender = st.selectbox("👥 Cinsiyet",
                                 ['Male', 'Female', 'Other'],
                                 help="Cinsiyet")
            gender_mapping = {
                'Male': 1,
                'Female': 0,
                'Other': -1
            }
            input_dict['gender'] = gender_mapping.get(gender, 1)
        
        st.divider()
        
        # ============================================================
        # KARİYER BİLGİLERİ
        # ============================================================
        st.markdown("**💼 Kariyer Bilgileri**")
        
        col1, col2 = st.columns(2)
        with col1:
            job_level = st.selectbox("📍 İş Seviyesi",
                                    ['Junior', 'Mid', 'Senior', 'Lead', 
                                     'Manager', 'Director', 'Executive'],
                                    help="Mevcut iş seviyesi")
            job_level_mapping = {
                'Junior': 1,
                'Mid': 2,
                'Senior': 3,
                'Lead': 4,
                'Manager': 5,
                'Director': 6,
                'Executive': 7
            }
            input_dict['job_level'] = job_level_mapping.get(job_level, 1)
        
        with col2:
            department = st.selectbox("🏢 Departman",
                                     ['Sales', 'IT', 'HR', 'Finance', 'Marketing',
                                      'Operations', 'R&D', 'Other'],
                                     help="Çalışılan departman")
            dept_mapping = {
                'Sales': 1, 'IT': 2, 'HR': 3, 'Finance': 4,
                'Marketing': 5, 'Operations': 6, 'R&D': 7, 'Other': 8
            }
            input_dict['department'] = dept_mapping.get(department, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            job_satisfaction = st.slider("😊 İş Memnuniyeti", 1, 5, 3)
            input_dict['job_satisfaction'] = job_satisfaction
        
        with col2:
            performance_rating = st.slider("⭐ Performans Notu", 1.0, 5.0, 3.5, 0.5)
            input_dict['performance_rating'] = performance_rating
        
        st.divider()
        
        # ============================================================
        # ŞİRKET BİLGİLERİ
        # ============================================================
        st.markdown("**🏛️ Şirket Bilgileri**")
        
        col1, col2 = st.columns(2)
        with col1:
            company_size = st.selectbox("📊 Şirket Büyüklüğü",
                                       ['Small', 'Medium', 'Large'],
                                       help="Şirketin büyüklüğü")
            size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
            input_dict['company_size'] = size_mapping.get(company_size, 2)
        
        with col2:
            industry = st.selectbox("🏭 Endüstri",
                                   ['Technology', 'Finance', 'Healthcare',
                                    'Retail', 'Manufacturing', 'Education', 'Other'],
                                   help="Çalışılan endüstri")
            industry_mapping = {
                'Technology': 1, 'Finance': 2, 'Healthcare': 3,
                'Retail': 4, 'Manufacturing': 5, 'Education': 6, 'Other': 7
            }
            input_dict['industry'] = industry_mapping.get(industry, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            bonus_eligible = st.checkbox("🎁 Bonus Hak Sahibi", value=True)
            input_dict['bonus_eligible'] = int(bonus_eligible)
        
        with col2:
            remote_work = st.selectbox("🏠 Uzaktan Çalışma",
                                      ['No', 'Hybrid', 'Full Remote'],
                                      help="Uzaktan çalışma durumu")
            remote_mapping = {'No': 0, 'Hybrid': 1, 'Full Remote': 2}
            input_dict['remote_work'] = remote_mapping.get(remote_work, 0)
        
        st.divider()
        
        # ============================================================
        # EKSİKLER
        # ============================================================
        st.markdown("**🎯 Ek Bilgiler**")
        
        col1, col2 = st.columns(2)
        with col1:
            certifications = st.slider("📜 Sertifikasyon Sayısı", 0, 10, 2)
            input_dict['certifications'] = certifications
        
        with col2:
            projects_completed = st.slider("✅ Tamamlanan Proje", 0, 50, 10)
            input_dict['projects_completed'] = projects_completed
        
        col1, col2 = st.columns(2)
        with col1:
            team_size = st.slider("👥 Yönetilen Takım Boyutu", 0, 100, 5)
            input_dict['team_size'] = team_size
        
        with col2:
            training_hours = st.slider("📚 Yıllık Eğitim Saati", 0, 200, 40)
            input_dict['training_hours'] = training_hours
        
        st.divider()
        predict_button = st.button("🔮 Maaş Tahmini Yap", use_container_width=True)
    
    # ============================================================
    # MAIN AREA - RESULTS
    # ============================================================
    if predict_button:
        with st.spinner('⏳ Maaş tahmini yapılıyor...'):
            # Tahmin yap
            predicted_salary = predict_salary(model, model_features, input_dict)
            
            if predicted_salary is not None and predicted_salary > 0:
                salary_category, salary_desc = get_salary_range_interpretation(predicted_salary)
                job_emoji = get_job_level_emoji(job_level)
                
                # Sonuç kutusu
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">💰 Tahmini Yıllık Maaş</div>
                    <div class="result-salary">${predicted_salary:,.0f}</div>
                    <div class="result-label">{salary_category} - {salary_desc}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✓ Tahmin başarıyla tamamlandı!")
                
                st.divider()
                
                # Metrikleri göster
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📅 Yaş", f"{age} yaş")
                with col2:
                    st.metric("⏱️ Tecrübe", f"{years_experience} yıl")
                with col3:
                    st.metric(f"{job_emoji} Seviye", job_level)
                with col4:
                    st.metric("🏢 Departman", department)
                
                st.divider()
                
                # Detaylı Bilgiler
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("👤 Kişisel Profil")
                    personal_df = pd.DataFrame({
                        'Özellik': ['Yaş', 'Eğitim', 'Cinsiyet', 'Tecrübe'],
                        'Değer': [f"{age} yaş", education_level, gender,
                                 f"{years_experience} yıl"]
                    })
                    st.table(personal_df)
                
                with col2:
                    st.subheader("💼 Kariyer Bilgileri")
                    career_df = pd.DataFrame({
                        'Özellik': ['Seviye', 'Departman', 'Memnuniyet', 'Performans'],
                        'Değer': [job_level, department, f"{job_satisfaction}/5",
                                 f"{performance_rating}/5.0"]
                    })
                    st.table(career_df)
                
                st.divider()
                
                # Maaş Analizi
                st.subheader("💵 Maaş Analizi")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    yearly = predicted_salary
                    st.metric("📊 Yıllık", f"${yearly:,.0f}")
                
                with col2:
                    monthly = yearly / 12
                    st.metric("📆 Aylık", f"${monthly:,.0f}")
                
                with col3:
                    if bonus_eligible:
                        bonus_estimate = yearly * 0.15  # %15 bonus
                        st.metric("🎁 Tahmini Bonus", f"${bonus_estimate:,.0f}")
                    else:
                        st.metric("🎁 Bonus", "Uygun Değil")
                
                st.divider()
                
                # Gauge Chart
                st.subheader("📊 Maaş Seviyesi")
                
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number",
                    value=predicted_salary,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Tahmini Yıllık Maaş ($)"},
                    gauge={
                        'axis': {'range': [0, max(200000, predicted_salary * 1.5)]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 30000], 'color': "#ecf0f1"},
                            {'range': [30000, 60000], 'color': "#bdc3c7"},
                            {'range': [60000, 100000], 'color': "#95a5a6"},
                            {'range': [100000, max(200000, predicted_salary * 1.5)], 'color': "#7f8c8d"}
                        ]
                    }
                )])
                
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Şirket Bilgileri
                st.subheader("🏛️ Şirket ve Endüstri")
                col1, col2 = st.columns(2)
                
                with col1:
                    company_df = pd.DataFrame({
                        'Özellik': ['Büyüklük', 'Endüstri', 'Uzaktan Çalışma'],
                        'Değer': [company_size, industry, remote_work]
                    })
                    st.table(company_df)
                
                with col2:
                    skills_df = pd.DataFrame({
                        'Özellik': ['Sertifikasyon', 'Proje', 'Takım Yönetimi', 'Eğitim Saati'],
                        'Değer': [f"{certifications} sertifika", 
                                 f"{projects_completed} proje",
                                 f"{team_size} kişi",
                                 f"{training_hours} saat"]
                    })
                    st.table(skills_df)
                
                st.divider()
                
                # Öneriler
                st.subheader("💡 Kariyer Önerileri")
                
                tips = []
                
                # Tecrübe önerileri
                if years_experience < 2:
                    tips.append("🌱 Giriş Seviyesi - Teknik beceri geliştirmeye odaklan")
                elif years_experience < 5:
                    tips.append("📈 Orta Tecrübe - Liderlik becerisi kazanmaya başla")
                else:
                    tips.append("⭐ Deneyimli - Uzmanlık alanında derinleş")
                
                # Eğitim önerileri
                if education_level == 'High School':
                    tips.append("🎓 Lisans derecesi alarak maaşını %20-30 artıra bilir")
                elif education_level == 'Bachelor':
                    tips.append("🎓 Master derecesi ile kariyer ilerleme şansı artar")
                
                # Performans önerileri
                if performance_rating >= 4.5:
                    tips.append("⭐ Mükemmel Performans - Promosyon için iyi fırsat")
                elif performance_rating < 3:
                    tips.append("📊 Performansı iyileştirmek maaş artışını etkileyebilir")
                
                # Sertifikasyon önerileri
                if certifications < 2:
                    tips.append("📜 Sertifikasyon almak maaş potansiyelini %10-20 artırabilir")
                
                # Takım yönetimi önerileri
                if team_size > 5:
                    tips.append("👥 Takım Liderliği - Yönetici pozisyonuna yükselebilirsin")
                
                # Bonus önerileri
                if bonus_eligible:
                    bonus = yearly * 0.15
                    tips.append(f"🎁 Bonus Fırsatı - Tahmini ${bonus:,.0f} bonus kazanabilirsin")
                
                for tip in tips:
                    st.info(tip)
                
                st.divider()
                
                # Maaş Karşılaştırması
                st.subheader("📊 Maaş Kategorileri")
                
                categories_data = {
                    'Kategori': ['Giriş', 'Orta', 'Üst', 'Executive'],
                    'Min': [20000, 40000, 80000, 150000],
                    'Maks': [40000, 80000, 150000, 500000]
                }
                
                categories_df = pd.DataFrame(categories_data)
                
                fig_cat = px.bar(
                    x=categories_df['Kategori'],
                    y=categories_df['Maks'],
                    title='Maaş Kategorileri',
                    labels={'Maks': 'Maksimum Maaş ($)', 'Kategori': 'Seviye'},
                    color=['#ecf0f1', '#bdc3c7', '#95a5a6', '#7f8c8d']
                )
                
                fig_cat.add_hline(
                    y=predicted_salary,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Sizin Tahminiz: ${predicted_salary:,.0f}",
                    annotation_position="right"
                )
                
                fig_cat.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_cat, use_container_width=True)
                
            else:
                st.error("❌ Tahmin başarısız")
                st.warning("⚠️ Lütfen tüm özellikleri doldur")

if __name__ == "__main__":
    main()