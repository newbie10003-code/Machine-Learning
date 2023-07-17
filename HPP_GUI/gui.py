import streamlit as st
import joblib
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('decisionTModel.pkl' , 'rb'))

availability = pickle.load(open('AvailabilityEncode.pkl' , 'rb'))
location = pickle.load(open('LocationEncode.pkl', 'rb'))
size = pickle.load(open('SizeEncode.pkl','rb'))
society = pickle.load(open('SocietyEncode.pkl','rb'))
society_values = pickle.load(open("Society.pkl", "rb"))

def areaConvert(a):
        if a == 'Plot  Area':
            return 1
        elif a == 'Carpet  Area':
            return 2
        elif a == 'Built-up  Area':
            return 3
        elif a == 'Super built-up  Area':
            return 4

def cleanSqft(s):
    if(type(s)==int):
        return s
    elif (s.isdigit()):
        return int(s)
    else:
        return 0

def cleanBath(b):
    return int(b)

def bal(b):
    return int(b)

st.header('Bangalore House Price Predictions')

area = st.selectbox("Area Type: ", options=['Super built-up  Area', 'Built-up  Area', 'Plot  Area', 'Carpet  Area'])

loc = st.selectbox("Location: ", options = ['Electronic City Phase II', 'Chikka Tirupathi',
       'Lingadheeranahalli', 'Whitefield', '7th Phase JP Nagar',
       'Sarjapur', 'Mysore Road', 'Bisuvanahalli',
       'Raja Rajeshwari Nagar', 'Ramakrishnappa Layout', 'Binny Pete',
       'Thanisandra', ' Thanisandra', 'Electronic City',
       'Ramagondanahalli', 'Yelahanka', 'Hebbal', 'Kanakpura Road',
       'Kundalahalli', 'Sarjapur  Road', 'Ganga Nagar', 'Doddathoguru',
       'Adarsh Nagar', 'Bhoganhalli', 'Lakshminarayana Pura',
       'Begur Road', 'Varthur', 'Gunjur', 'Hegde Nagar', 'Haralur Road',
       'Hennur Road', 'Cholanayakanahalli', 'Kodanda Reddy Layout',
       'EPIP Zone', 'Dasanapura', 'Kasavanhalli', 'Sanjay nagar',
       'Kengeri', 'Yeshwanthpur', 'Chandapura', 'Kothanur',
       'Green View Layout', 'Shantiniketan Layout', 'Rajaji Nagar',
       'Devanahalli', 'Byatarayanapura', 'Akshaya Nagar',
       'LB Shastri Nagar', 'Hormavu', 'Peenya', 'Kudlu Gate',
       '8th Phase JP Nagar', 'Chandra Layout', 'Anandapura',
       'Kengeri Satellite Town', 'Basavanapura', 'Kannamangala',
       'Hulimavu', 'Hosa Road', 'Keshava Nagar', 'RMV Extension',
       'Tejaswini Nagar', 'Jai Bheema Nagar', 'Attibele','CV Raman Nagar', 'Malleshwaram', 'Hebbal Kempapura',
       'Vijayanagar', 'KR Puram', 'Marathahalli', 'Pattandur Agrahara',
       'HSR Layout', 'Kadugodi', 'Kogilu', 'Panathur', 'Kammasandra',
       'Electronics City Phase 1', 'Tala Cauvery Layout', 'Dasarahalli',
       'Koramangala', 'Muthurayya Swamy Layout', 'Budigere',
       'Dodda Nekkundi Extension', 'Mylasandra', 'Kalyan nagar',
       'Ashwath Nagar', 'Ncpr Industrial Layout', 'Meenakunte',
       'OMBR Layout', 'Coffee Board Layout', 'Ambedkar Nagar',
       'Geleyara Balaga Layout', 'Kalena Agrahara', 'Talaghattapura',
       'Balagere', 'Jigani', 'Gollarapalya Hosahalli', 'Old Madras Road',
       'Silver Springs Layout', '9th Phase JP Nagar', 'Jakkur',
       'Maruthi Sevanagar', 'RMV 2nd Stage', 'Singasandra', 'AECS Layout',
       'Mallasandra', 'Begur', 'JP Nagar', 'Sunder Ram Shetty Nagar',
       'Motappa Layout', 'Kaval Byrasandra', 'Kaggalipura',
       'Basavanna Nagar', '6th Phase JP Nagar', 'Ulsoor', 'Uttarahalli',
       'Thigalarapalya', ' Devarachikkanahalli', 'Bommasandra',
       'Prashanth Nagar', 'Suragajakkanahalli', 'Ardendale', 'Harlur',
       'Sampigehalli', 'Kodihalli', 'Magadi Road', 'Narayanapura',
       'Hennur', '5th Phase JP Nagar', 'Kodigehaali', 'Bannerghatta Road','Gopalapura', 'Billekahalli', 'Jalahalli', 'Sompura',
       'Ashirvad Colony', 'Dodda Nekkundi', 'Hosur Road', 'Amco Colony',
       'Ambalipura', 'Hoodi', 'Samethanahalli', 'Brookefield',
       'Suddaguntepalya', 'Udayapur Village', 'Bellandur', 'Vittasandra',
       'Giri Nagar', 'Chikkabidarakallu', '1 Giri Nagar', 'Hoysalanagar',
       'Defence Colony', 'Amruthahalli', 'Patelappa Layout',
       'Subramanyapura', '3rd Block Hrbr Layout', 'Surabhi Layout',
       'Omkar Nagar', 'Kambipura', 'VHBCS Layout', 'Rajiv Nagar',
       'Gattahalli', 'Arekere', 'Mico Layout', 'Munnekollal',
       'Banashankari Stage III', 'Dooravani Nagar', 'JCR Layout',
       'Nehru Nagar', 'Sneha Colony', 'Konanakunte', 'Ashwini layout',
       'Gottigere', 'HRBR Layout', 'Kanakapura', 'Tumkur Road',
       'Hosahalli', 'Jalahalli West', 'GM Palaya', 'Jalahalli East',
       'Hosakerehalli', 'Nagondanahalli', 'Shanthala Nagar',
       'Bettahalsoor', 'Ambedkar Colony', 'Avalahalli', 'Prakruthi Nagar',
       'Abbigere', 'Tindlu', 'Green Garden Layout', 'Gubbalala',
       'Dairy Circle', 'Narayana Nagar 1st Block', 'KSRTC Layout',
       'New Gurappana Palya', 'Palanahalli', 'Vadarpalya', 'Kudlu',
       'Old Airport Road', 'Vishwapriya Layout', 'Banashankari Stage VI','Battarahalli', 'HMT Layout', 'Kaggadasapura', 'ITI Layout',
       'Yelahanka New Town', 'Sahakara Nagar', 'Rachenahalli',
       'Kodbisanhalli', 'Kodichikkanahalli', 'Bendiganahalli',
       'Ferrar Nagar', 'Green Glen Layout', 'M.G Road',
       'Horamavu Banaswadi', '1st Phase JP Nagar', 'Kaverappa Layout',
       'Devarabisanahalli', 'Somasundara Palya', 'Vidyaranyapura',
       'Babusapalaya', 'Nagappa Reddy Layout', 'TC Palaya',
       'Suraksha Nagar', 'Iblur Village', 'Yelachenahalli',
       'Basava Nagar', '2nd Block Hrbr Layout', 'Basapura',
       'Channasandra', 'Singena Agrahara', 'Mango Garden Layout',
       'Choodasandra', 'Indira Nagar', 'Sai Gardens', 'Mahadevpura',
       'Hanumanth Nagar', 'Basaveshwara Nagar', 'Kaikondrahalli',
       'Hunasamaranahalli', 'RWF West Colony', 'Bileshivale',
       'Neeladri Nagar', 'Frazer Town', 'Jaya Nagar East', 'Iggalur',
       'Banashankari', 'Chamrajpet', 'VGP Layout', 'Vasanth nagar',
       'Kalkere', 'Siddapura', 'Maragondanahalli', 'Ramamurthy Nagar',
       'Garudachar Palya', 'Roopena Agrahara', 'Gollahalli',
       'Sonnenahalli', 'D Souza Layout', 'Nagarbhavi', 'Bommanahalli',
       'Chikkalasandra', 'Dommasandra', 'Byadarahalli', 'Judicial Layout',
       'Outer Ring Road East', 'Vinayaka Nagar', 'GB Palya','Ashwathnagar', 'Kasturi Nagar', 'Belathur', 'Srirampura',
       'Devanahalli Road', 'Ejipura', 'Green Woods Layout',
       'Craig Park Layout', 'Immadihalli', 'Muneshwara Nagar',
       'Rayasandra', 'Malleshpalya', 'Parappana Agrahara',
       'Lakshmi Layout', 'Thirumenahalli', 'KPC Layout',
       'Daadys Gaarden Layout', 'Kothannur', 'Marsur', 'Karuna Nagar',
       'Kallumantapa', 'Malimakanapura', 'Medahalli',
       'Rustam Bagh Layout', 'Garden Layout', 'T K Reddy Layout',
       'Doddanekundi', 'Venkatadri Layout', 'Bommenahalli',
       'Mahaganapathy Nagar', 'HBR Layout', 'Vittal Nagar',
       'Bhuvaneshwari Nagar', 'Prithvi Layout', 'Domlur', 'Thubarahalli',
       'Jaya Mahal layout', 'BSM Extension', 'Vijinapura', 'Byrasandra',
       'Chowdeshwari Layout', 'Sector 2 HSR Layout', 'Padmanabhanagar',
       'Badavala Nagar', '4th Block Koramangala', 'Belatur',
       'Nallurhalli', 'Kereguddadahalli', 'Laxmi Sagar Layout',
       'Bannerghatta', 'Harappanahalli', 'BTM Layout', 'Kanaka Nagar',
       'NR Colony', 'Byagadadhenahalli', 'Doddabommasandra',
       'Sarjapura - Attibele Road', 'Maruthi Layout',
       'Sree Narayana Nagar', 'Tunganagara', 'Nagavara','Remco Bhel Layout', 'Chokkasandra', 'Panduranga Nagar',
       'Jakkur Plantation', '1st Block Koramangala',
       'Shree Ananth Nagar Layout', 'Hoskote', 'Sector 1 HSR Layout',
       'BTM 2nd Stage', 'Ananth Nagar', 'Sundar Ram Shetty Nagar',
       'Alfa Garden Layout', 'Hoodi Layout', 'Seegehalli',
       'Gaundanapalya', '2nd Phase JP Nagar', 'Doctors Layout',
       'Basavangudi', 'Vishwapriya Nagar', 'Sarakki Nagar', 'R.T. Nagar',
       'Sector 7 HSR Layout', 'Hennur Gardens',
       'Howthinarayanappa Garden', 'Bharathi Nagar', 'Cambridge Layout',
       'Doddakannelli', 'Cox Town', 'Pulkeshi Nagar', 'Jayanagar',
       ' Bhoganhalli', 'Pai Layout', '8th block Koramangala', 'Bidadi',
       'Amruthnagar', 'Sathya Sai Layout', 'Rajiv Gandhi Nagar', 'Anekal',
       'Bhagyalakshmi Avenue', 'Doddaballapur', 'Horamavu Agara',
       'Chinnapanahalli', 'Balaji Gardens Layout', ' Rachenahalli',
       'Akshaya Vana', 'Channasandra Layout', 'Gopalkrishna Nagar',
       'Volagerekallahalli', 'Keerthi Layout', 'Shikaripalya', 'Hagadur',
       'Soundarya Layout', 'Cunningham Road', 'Dollars Layout',
       'Nagavarapalya', 'Sultan Palaya', 'Gopal Reddy Layout',
       'Thurahalli', 'Murugeshpalya', 'Kadubeesanahalli',
       'Cleveland Town', 'Kada Agrahara', 'Bellari Road',
       'Abbaiah Reddy Layout', 'Tata Nagar', ' Devarabeesana Halli',
       'Brindavan Nagar', 'Seetharampalya', 'B Narayanapura',
       'Raghuvanahalli', 'Wilson Garden', 'Challaghatta', 'KR Garden',
       'Kathriguppe', 'Sahyadri Layout', 'Bagalur', 'P Krishnappa Layout',
       'Crimson Layout', 'Kyalasanahalli', 'Ckikkakammana Halli',
       'Munivenkatppa Layout', 'Vijaya Bank Layout', 'Kumarapalli',
       'Lingarajapuram', 'Vasantha Vallabha Nagar', 'Kalhalli',
       'Kumaraswami Layout', 'Hadosiddapura', 'Kachanayakanahalli',
       'Yelenahalli', '6th block Koramangala', 'Vignana Nagar',
       'Canara Bank Colony', 'Hanumagiri', 'Benson Town',
       'Akshayanagara West', 'OLd Gurappanapalya', 'Ramanjaneyanagar',
       'B Channasandra', 'New Thippasandra', 'Gattigere', 'Kamakshipalya',
       'Dollars Colony', 'Pragathi Nagar', 'HAL 2nd Stage',
       'Poorna Pragna Layout', 'Kenchenahalli', 'Kundalahalli Colony',
       'Amblipura', 'Vimanapura', 'Dodsworth Layout', 'Devasthanagalu',
       'Venugopal Reddy Layout', 'Lake City', 'S R Layout',
       'Sadduguntepalya', 'Mallathahalli', 'Doddakammanahalli',
       'Chikkathoguru', 'Richards Town', 'BCC Layout', 'Vinayak Nagar',
       'Jaladarsini Layout', 'Brooke Bond First Cross',
       'Glass Factory Layout', 'Raghavendra Nagar', 'Bhuvaneswari Nagar',
       ' Electronic City', 'Mathikere', 'Kempegowda Nagar',
       'Nyanappana Halli', 'Nelamangala', ' Whitefield',
       'Vaishnavi Layout', 'Naganathapura', 'Venkatapura',
       '3rd Phase JP Nagar', 'Devarachikkanahalli', 'Anjanapura',
       'Reliaable Tranquil Layout', 'Mailasandra', 'Chelekare',
       'Doopanahalli', 'Kattigenahalli', 'Varthur Road', 'Bethel Nagar',
       'Dollar Scheme Colony', 'Omarbagh Layout',
       'Aishwarya Crystal Layout', '2nd Block Jayanagar',
       'Nayandanahalli', 'DUO Layout', 'ISRO Layout', 'Hennur Bande',
       'Shampura', 'Langford Gardens', 'Richmond Town',
       'Vishwanatha Nagenahalli', 'Friends Colony', 'Shettigere',
       'Rahat Bagh', 'NTI Layout', 'Gunjur Palya', 'Maithri Layout',
       'Nobo Nagar', 'Krishna Reddy Layout', 'Kuvempu Layout',
       'Kaveri Nagar', 'Devi Nagar', 'Chikkabanavar',
       'Bommasandra Industrial Area', 'Stage-4 Bommanahalli',
       'Banashankari Stage V', 'Manorayana Palya', 'Sathya Layout',
       'Shanti Nagar', 'Jayamahal', 'Veer Sandra', 'Chennappa Layout',
       'Jeevan bima nagar', 'Lavakusha Nagar', 'Vittal Mallya Road',
       'Dena Bank Colony', 'Doddabidrakallu', 'NGR Layout',
       'Jnana Ganga Nagar', 'Ittamadu', 'Chokkanahalli', 'Vikram Nagar',
       'Sarvabhouma Nagar', 'Coconut Grove Layout', 'Kadabagere',
       'Anand Nagar', 'Kammanahalli', 'Janatha Colony',
       '1st Block Jayanagar', 'Veersandra', 'Kumbena Agrahara',
       'Hiremath Layout', 'Nagadevanahalli', 'Yemlur',
       'Ramamurthy Nagar Extension', 'Himagiri Meadows',
       'Narayanappa Garden', 'Kempapura', 'Carmelaram', 'Ankappa Layout',
       'Jakkuru Layout', 'Maruthi Nagar', 'Punappa Layout',
       'Pampa Extension', 'Banaswadi', 'Sanne Amanikere',
       '2nd Stage Arekere Mico Layout', 'BCMC Layout', 'Uday Nagar',
       'Chaitanya Ananya', 'Kacharakanahalli', 'Tigalarpalya',
       ' Banaswadi', 'Sugama Layout', 'Hongasandra', 'Anugrah Layout',
       'Bandepalya', 'AMS Layout', 'RPC layout', 'Gulimangala',
       'Addischetan Layout', 'Sadaramangala', 'Konena Agrahara',
       'Raghavendra Layout', 'Varsova Layout',
       'Raja Rajeshwari Nagar 5th Stage', '7th Block Jayanagar',
       'Bande Nallasandra', 'Upkar Layout', 'Singapura Village',
       'Canara Bank Layout', 'Queens Road', 'Hosapalya', 'Jagadish Nagar',
       'Rajasree Layout', 'Chikkadunnasandra', 'Thirumalashettyhally',
       'Vibuthipura', 'Kodigehalli', 'P&T Layout', 'Varanasi',
       'Virudhu Nagar', 'Ramanashree Enclave', 'Chikku Lakshmaiah Layout',
       'Shivaji Nagar', 'Chikkagubbi', 'Kammagondahalli',
       'Yeshwanthpur Industrial Suburb', 'Thirupalya', 'Tirumanahalli',
       'ITPL', 'NRI Layout', 'Adugodi', 'Sidedahalli', 'Shanthi Pura',
       'Anwar Layout', 'Doddanakundi Industrial Area 2', 'RBI Layout',
       'Shauhardha Layout', 'Bhoopsandra', '3rd Block Koramangala',
       'Kuvempu Nagar', 'Sri Balaji Krupa Layout', "St. John's Road",
       'Govindapura', 'Ayappa Nagar', 'Hessarghatta', 'Anantapuram',
       'Chikka Banaswadi', 'Akshayanagara East', 'Guddadahalli',
       'Kashi Nagar', 'Nanjappa Garden', 'Allalasandra',
       'Maheswari Nagar', 'Udayagiri', 'Mahalakshmi Layout',
       'Rahmath Nagar', 'Halanayakanahalli', 'Air View Colony',
       'Amam Enclave Layout', 'Anantapura', 'Mallappa Layout',
       'Devarabeesana Halli', 'Chikbasavanapura', 'Billamaranahalli',
       'Jogupalya', 'Gulakamale', 'Mahalakshmi Puram', 'Jakkasandra',
       'Cooke Town', 'Virupakshapura', 'Kenchenhalli', 'Church Street',
       'Sarvobhogam Nagar', 'Near International Airport', 'Palace Road',
       'Abshot Layout'])


sizebhk = st.selectbox("Size: ", options=['2 BHK', '4 Bedroom', '3 BHK', '3 Bedroom', '1 RK', '4 BHK',
       '1 BHK', '5 BHK', '11 BHK', '5 Bedroom', '9 BHK', '2 Bedroom',
       '6 BHK', '7 BHK', '6 Bedroom'])


soc = st.selectbox("Society: ",options = society_values)

avail = st.selectbox("Availability: ", options = ['19-Dec', 'Ready To Move', '18-Nov', '17-Oct', '21-Dec', '19-Sep',
       '20-Sep', '18-Mar', '18-Apr', '20-Aug', '19-Mar', '17-Sep',
       '17-Aug', '19-Apr', '22-Dec', '18-Aug', '19-Jan', '17-Jul',
       '18-Jul', '18-May', '18-Dec', '21-Jun', '18-Sep', '17-May',
       '17-Jun', '18-Oct', '21-May', '20-Dec', '18-Jun', '16-Mar',
       '22-Jun', '17-Dec', '21-Feb', '19-May', '17-Nov', '20-Oct',
       '20-Jun', '18-Feb', '19-Feb', '21-Oct', '21-Jan', '17-Mar',
       '19-Jun', '17-Apr', '22-May', '19-Oct', '21-Jul', '21-Nov',
       '21-Mar', '19-Jul', '20-Jan', '21-Sep', '18-Jan', '20-Mar',
       '19-Nov', '15-Jun', '19-Aug', '20-May', '20-Nov', '20-Jul',
       '20-Feb', '15-Dec', '21-Aug', '16-Oct', '22-Nov', '16-Dec',
       '15-Aug', '17-Jan', '16-Nov', '20-Apr', '22-Jan', '16-Jan',
       '17-Feb', '14-Jul'])


bathroom = st.select_slider("Number of Bathrooms: ",options=[2., 5., 3., 4., 1., 6., 9., 7.])

balcony = st.select_slider("Number of balconies: ",options = [1., 3., 2., 0.])

total_sqft = st.slider("sqft",  0 ,4689)

area = areaConvert(area)
avail=availability.transform([avail])[0]
loc = location.transform([loc])[0]
size=size.transform([sizebhk])[0]
soc=society.transform([soc])[0]
sqft = cleanSqft(total_sqft)
bath = cleanBath(bathroom)
balcony = bal(balcony)

p=[[area,avail,loc,size,soc,sqft,bath,balcony]]

st.write("Predicted Price is Rs ")

ans = model.predict(p)

st.write("The predicted value of the house is: ", ans[0])