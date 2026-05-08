import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import psycopg2
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(debug=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

with open('./ModelSaving/model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Inside model")
scalar = pickle.load(open('./Scaler/Scalar.pkl', 'rb'))
print("inside scalar")

try:
    DATABASE_URL = os.getenv("DATABASE_URL")
    db = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = db.cursor()
    print("DB Connected")
    
    cur.execute("CREATE TABLE IF NOT EXISTS incomecensus4(Age int, FinalWeight int, MaxEducationalQualification varchar(20), CapitalGain varchar(5), CapitalLoss varchar(5), WorkSector varchar(20), MaritalStatus varchar(15),"
            "Race varchar(15), Gender varchar(15), WorkStyle varchar(20), Country varchar(10))")
    db.commit()
except Exception as e:
    print("DB Connection Failed")
    print(e)
    cur = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    print("Inside home page")
    return templates.TemplateResponse(request=request, name='home.html', context={"request": request})

@app.get("/info", response_class=HTMLResponse)
async def info(request: Request):
    print("Inside info page")
    return templates.TemplateResponse(request=request, name='info.html', context={"request": request})

@app.get("/developer", response_class=HTMLResponse)
async def developer(request: Request):
    print("Inside home page")
    return templates.TemplateResponse(request=request, name='developer.html', context= {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    print("Inside contact page")
    return templates.TemplateResponse(request=request, name='contact.html', context= {"request": request})

@app.get("/app", response_class=HTMLResponse)
async def index_page(request: Request):
    print("Inside app")
    return templates.TemplateResponse(request=request, name='index.html', context= {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    Final_Weight: int = Form(...),
    Education: str = Form(...),
    CapitalGain: str = Form(...),
    CapitalLoss: str = Form(...),
    WorkClass: str = Form(...),
    MaritalStatus: str = Form(...),
    race: str = Form(...),
    gender: str = Form(...),
    hours: str = Form(...),
    country: str = Form(...)
):
    print("🔥 PREDICT HIT", flush=True) 
    wt = Final_Weight

    edu = Education
    if edu == 'Higher Studies':
        edu = 0
    elif edu == 'Bachelors':
        edu = 1
    elif edu == 'Associate':
        edu = 2
    elif edu == 'Prof-School':
        edu = 3
    elif edu == 'Diploma':
        edu = 4
    else:
        edu = 5

    gain = CapitalGain
    if gain == 'Yes':
        gain = 1
    else:
        gain = 0

    loss = CapitalLoss
    if loss == 'Yes':
        loss = 0
    else:
        loss = 1

    wrk_cls = WorkClass
    if wrk_cls == 'Private':
        wrk_cls = 0, 0, 0
    elif wrk_cls == 'Government':
        wrk_cls = 0, 0, 1
    elif wrk_cls == 'SelfEmployeed':
        wrk_cls = 0, 1, 0
    else:
        wrk_cls = 1, 0, 0

    status = MaritalStatus
    if status == 'Married':
        status = 0
    else:
        status = 1

    race_val = race
    if race_val == 'White':
        race_val = 0, 0
    elif race_val == 'Brown':
        race_val = 1, 0
    else:
        race_val = 0, 1

    gen = gender
    if gen == 'Male':
        gen = 1
    else:
        gen = 0

    hrs = hours
    if hrs == 'ideal':
        hrs = 0, 0
    elif hrs == 'over':
        hrs = 1, 0
    else:
        hrs = 0, 1

    country_val = country
    if country_val == 'US':
        country_val = 1
    else:
        country_val = 0

    col = ([[age, wt, edu, gain, loss, *wrk_cls, status, *race_val, gen, *hrs, country_val]])
    print(col)

    # scaled_col = scalar.transform(col)
    # print(scaled_col)

    # prediction = model.predict(scaled_col)
    # print(prediction)

    print("RAW INPUT:", col, flush=True)
    print("TOTAL FEATURES:", len(col[0]), flush=True)

    scaled_col = scalar.transform(col)
    print("SCALED INPUT:", scaled_col, flush=True)

    prediction = model.predict(scaled_col)
    print("PREDICTION:", prediction, flush=True)

    if cur and db:
        cur.execute(
            "INSERT INTO incomecensus4 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (age, wt, Education, CapitalGain, CapitalLoss, WorkClass, MaritalStatus, race, gender, hours, country))
        db.commit()
        # CHECK DATA FROM DB
        cur.execute("SELECT * FROM incomecensus4 ORDER BY Age DESC LIMIT 5")  
        rows = cur.fetchall()
        
        print("LATEST DATA FROM DB:", flush=True)
        for r in rows:
            print(r, flush=True)
    else:
        print("No DB connection, skipping data insertion.")

    if (prediction == np.array(1)).all():
        return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "request": request,
            "Prediction_text": "The Salary of an Individual is More than 50K"
        }
    )
        
    else:
        return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "request": request,
            "Prediction_text": "The Salary of an Individual is Less than 50K"
        }
    )

@app.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    return templates.TemplateResponse(request=request, name='home.html', context={"request": request})


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)