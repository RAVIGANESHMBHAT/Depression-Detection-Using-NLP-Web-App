from flask import Flask, request,  render_template,redirect

import numpy as np
import csv
import time
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D

import speech_recognition as sr
from googletrans import Translator
 
import pytesseract        
# adds image processing capabilities 
from PIL import Image 

translator = Translator()

SENTIMENT_THRESHOLDS = (0.4, 0.6)
text=""      # globally declared variable which holds the user entered sentence
label=""     # depression status POSITIVE/NEUTRAL/NEGATIVE
language=""  #it holds the language of user entered sentence
all_solutions=""
translate_link=""
LANGUAGES = {
    'ar': 'arabic',
    'zh-cn': 'chinese (simplified)',
    'en': 'english',
    'de': 'german',
    'gu': 'gujarati',
    'hi': 'hindi',
    'ja': 'japanese',
    'kn': 'kannada',
    'ko': 'korean',
    'ml': 'malayalam',
    'mr': 'marathi',
    'ne': 'nepali',
    'pa': 'punjabi',
    'ru': 'russian',
    'es': 'spanish',
    'ta': 'tamil',
    'te': 'telugu',
    'ur': 'urdu'
}


'''with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)'''
tokenizer=pickle.load(open('tokenizer.pkl', 'rb'))
    
vocab_size = len(tokenizer.word_index) + 1
w2v_model=pickle.load(open('w2v_model2.pkl', 'rb'))
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)


'''model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.1))
model.add(LSTM(100, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(1, activation='sigmoid'))'''
model = Sequential()
# Embedded layer
model.add(Embedding(len(embedding_matrix), 300, weights=[embedding_matrix], input_length=300, trainable=False))
# Convolutional Layer
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
# LSTM Layer
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

model.load_weights('model_depression.h5')
print("The model summary is:")
print(model.summary())

app = Flask(__name__)
#model1 = pickle.load(open('finalized_model2.pkl', 'rb'))

def decode_sentiment(score):       
    if score <= 0.429:
        label = 'NEGATIVE'
    elif score >= 0.6:
        label = 'POSITIVE'
    else:
        label = 'NEUTRAL'
    return label

def predict1(text):
    global label
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
    # Predict
    score = model.predict([x_test])
    print("Score= ",score)
    # Decode sentiment
    label = decode_sentiment(score[0])
    return label,float(score),time.time()-start_at

@app.route('/')
def home():
    return render_template('front.html')
@app.route('/home1')
def home1():
    return render_template('front1.html')


@app.route('/predict',methods=['POST'])
def predict():
    global text,language
    print("Starting of prediction")
    text=request.form['depress']
    print(text)
    
    if text!="Sorry..! Cannot recognize the speech.           Check the internet connection." and text!="Recognizing..." and text!=" ":
        try:
            language=translator.detect(text).lang
            text1=translator.translate(text, dest='en')
            text2=translator.translate(text,dest=language)
            print(language)
            print(text1.text)
            
            k,l,m=predict1(text1.text)
            s="Status : "+text2.text
            s1="Result : "+k
            if 'actual' in request.form:
                if s1=="Result : NEGATIVE":
                    return render_template('index1.html', prediction_text=s,prediction_text1=s1)#'Status     : {} \n Result : {}'.format(text,k))
                else:
                    if s1=="Result : POSITIVE":
                        emoji_path="static\images\happy.jpg"
                    else:
                        emoji_path="static\images\happy1.jpg"
                    return render_template('index.html', prediction_text=s,prediction_text1=s1, emoji=emoji_path)
            elif 'positive' in request.form:
                return render_template('index.html', prediction_text=s,prediction_text1="Positive", emoji="static\images\happy.jpg")
            else:
                return render_template('index1.html', prediction_text=s,prediction_text1="Negative")
        except:
            print("No internet connection..!")
            return render_template('front.html')
    else:
        return render_template('front.html')

@app.route('/solutions',methods=['POST'])
def solutions():
    global text,label,language,sentence_button,all_solutions,translate_link
    text1=translator.translate(text, dest='en')
    text=text1.text.lower()
    print("Solution text= ",text)
    c=0     # variable count, which keeps track of the number of solutions displayed
    flag=0  #just to check whether any solution is displayed or not
    all_solutions=""
    
    with open('solution/Depr_solution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if set(row[0].split()) & set(text.split()):
                c+=1
                if c<5:
                    all_solutions+=row[1]    
                    flag=1
        if flag==0 and label=='NEGATIVE':
            all_solutions="Keep calm first..Just look into the way in which you have to go in order to achieve your goal.Do meditation and do the work which gives you the happyness.That's it. When God takes out the trash, don't go digging back through it. Trust Him."
    #all_solutions=translator.translate(all_solutions, dest=language).text
    sentence_button=translator.translate("Click to see translated solution", dest=language)
    mmm=sentence_button.text
    
    translate_link = "https://translate.google.co.in/#view=home&op=translate&sl=en&tl=" + language + '&text='
    l=all_solutions.split(' ')
    for i in range(len(l)-1):
        translate_link = translate_link + l[i] + '%20'
    translate_link = translate_link + l[len(l)-1]
    return render_template('solution.html', soln=all_solutions,languag=mmm)

@app.route('/speech2text',methods=['POST'])
def speech2text():
    languages=request.form.get('abc')  # the language which is selected from the drop-down
    previous_content=request.form.get('abcd')
    r=sr.Recognizer()
    with sr.Microphone() as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source, duration=4)
        # recognize (convert from speech to text)
        try:
            texts = r.recognize_google(audio_data,language=languages)  #,language='kn-IN'
            if texts=="clear":
                return render_template('front.html',converted_text="")
            if previous_content!="Sorry..! Cannot recognize the speech.           Check the internet connection.":
                texts=previous_content+' '+texts
        except:
            texts="Sorry..! Cannot recognize the speech.           Check the internet connection." 
               
        print("Text= ",texts)
    return render_template('front.html',converted_text=texts)
    
@app.route('/image_predict',methods=['POST'])
def image_predict():
    languages=request.form.get('abc')
    global text,language
    name1 = request.files['file'].filename
    path="static/images/Images of depression/"+name1
    img = Image.open(path)
    pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'   
    # converts the image to result and saves it into result variable 
    text = pytesseract.image_to_string(img,lang=languages)  #,lang='kan'

    print("text1 in imagepredict= ",text)
    try:
        language=translator.detect(text).lang
        text1=translator.translate(text, dest='en')
        text2=translator.translate(text,dest=language)
        print(language)
        print(text1.text)
            
        k,l,m=predict1(text1.text)
        s="Status : "+text2.text
        s1="Result : "+k
            
        if s1=="Result : NEGATIVE":
            return render_template('index1.html', prediction_text=s,prediction_text1=s1)#'Status     : {} \n Result : {}'.format(text,k))
        else:
            return render_template('index.html', prediction_text=s,prediction_text1=s1)
    except:
        print("No internet connection..!")
        return render_template('front1.html')
    
    
    #return render_template('front1.html',text=result,res=k)

@app.route('/webpage',methods=['GET','POST'])
def webpage():
    return render_template('Depression solution website/home.html')
@app.route('/ind1',methods=['GET','POST'])
def ind1():
    return render_template('Depression solution website/in1.html')

@app.route('/ind2',methods=['GET','POST'])
def ind2():
    return render_template('Depression solution website/in2.html')
@app.route('/ind3',methods=['GET','POST'])
def ind3():
    return render_template('Depression solution website/in3.html')
@app.route('/ind4',methods=['GET','POST'])
def ind4():
    return render_template('Depression solution website/in4.html')
@app.route('/ind5',methods=['GET','POST'])
def ind5():
    return render_template('Depression solution website/in5.html')
@app.route('/ind6',methods=['GET','POST'])
def ind6():
    return render_template('Depression solution website/in6.html')
@app.route('/ind7',methods=['GET','POST'])
def ind7():
    return render_template('Depression solution website/in7.html')
@app.route('/ind8',methods=['GET','POST'])
def ind8():
    return render_template('Depression solution website/in8.html')
@app.route('/ind9',methods=['GET','POST'])
def ind9():
    return render_template('Depression solution website/in9.html')

@app.route('/google_translator',methods=['GET','POST'])
def google_translator():
    return redirect(translate_link)

@app.route('/webForamTeam',methods=['GET','POST'])
def webForamTeam():
    return render_template('index4.html')

if __name__ == "__main__":
    app.run(debug=True)