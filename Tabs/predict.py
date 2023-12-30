import streamlit as st 

from web_function import predict

def app(df,x,y):
    
    st.title("Halaman Prediksi")
    
    Age = st.text_input('Input Umur')
    select_EmploymentType = st.selectbox('Pilih Ketenagaan Kerja',['Government Sector','Private Sector/Self'])
    
    if select_EmploymentType == 'Government Sector':
        EmploymentType = 1
    else :
        EmploymentType = 0
        
    select_GraduateOrNot =st.selectbox('Apakah Pelanggan Sudah Lulus kuliah ?',['Yes','No'])
    
    if select_GraduateOrNot == 'Yes':
        GraduateOrNot = 1
    else:
        GraduateOrNot = 0
    select_AnnualIncome =st.selectbox('Input Pendapatan Tahunan ',['Pendapatan Rendah','Pendapatan Sedang','Pendapatan Tinggi'])
        
    if select_AnnualIncome == 'Pendapatan Rendah':
        AnnualIncome = 1
    elif select_AnnualIncome == 'Pendapatan Sedang':
        AnnualIncome = 2
    else:
        AnnualIncome = 3
    st.text("Keterangan : < 800000 = Pendapatan rendah, >= 800000 dan <=1300000 = Pendapatan Sedang, > 1300000 = Pendapatan Tinggi") 
    FamilyMembers =st.slider('Input Jumlah Anggota keluarga',min_value=1,max_value=6)
    select_ChronicDiseases = st.selectbox('Apakah Pelanggan Menderita Penyakit Atau Kondisi Besar Seperti Diabetes, Asma, dll. ?',['Yes','No'])
    
    if select_ChronicDiseases == 'Yes':
        ChronicDiseases = 1
    else:
        ChronicDiseases = 0
    select_FrequentFlyer =st.selectbox('Apakah Pelanggan Sudah sering melakukan penerbangan?',['Yes','No'])
    
    if select_FrequentFlyer == 'Yes':
        FrequentFlyer = 1
    else:
        FrequentFlyer = 0
    select_EverTravelledAbroad =st.selectbox('Apakah Pelanggan Pernah Bepergian ke Luar Negeri?',['Yes','No'])
    
    if select_EverTravelledAbroad == 'Yes':
        EverTravelledAbroad = 1
    else:
        EverTravelledAbroad = 0
    
    features = [Age,EmploymentType,GraduateOrNot,AnnualIncome,FamilyMembers,ChronicDiseases,FrequentFlyer,EverTravelledAbroad]
    
    if st.button("Prediksi"):
        prediction, score = predict(x,y,features)
        score = score
        st.info("Prediksi Sukses...")
        
        if (prediction == 1):
            st.success ("Pelanggan tersebut kemungkinan besar membeli paket asuransi perjalanan")
        else:
            st.warning("Pelanggan tersebut kemungkinan besar tidak membeli paket asuransi perjalanan")
    
        st.write("Model yang di gunakan memiliki tingkat akurasi ", (score*100),"%")