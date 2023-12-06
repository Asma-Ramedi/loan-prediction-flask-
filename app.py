from flask import Flask , request , render_template
import pickle
import pandas as pd
import numpy as np

# on a des variables, on les a importés
def load(path):
    with open(path,'rb') as f : 
        var = pickle.load(f)
    return(var)

le_Education = load('./Education.pickle')
le_Gender = load('./Gender.pickle')
le_Loan_Status = load('./Loan_Status.pickle')
le_Married = load('./Married.pickle')
le_Property_Area = load('./Property_Area.pickle')
le_Self_Employed = load('./Self_Employed.pickle')
model = load('./model.pickle')



# flask app 


# to run this program open your CMD.exe and cd into the folder 
# type in your cmd :   python app.py     
# open your browser and open 127.0.0.1:5000






app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def intro():
    # si on est dans un état de soumission(POST)
    if request.method =="POST":
        if request.form["Gender"] and request.form["Married"] and request.form["Dependents"] and request.form["Education"]:
            # on récupère les données à partir de HTML , HTML-->python
            x = pd.DataFrame(columns = ["Gender"])
            x["Gender"] = le_Gender.transform([request.form["Gender"]])
            x["Married"] = le_Married.transform([request.form["Married"]])
            x['Dependents'] = float(request.form["Dependents"])
            x['Education'] = le_Education.transform([request.form["Education"]])
            x["Self_Employed"] = le_Self_Employed.transform([request.form["Self_Employed"]])
            ApplicantIncome = float(request.form["ApplicantIncome"])
            x['Credit_History'] = float(request.form["Credit_History"])
            x['Property_Area'] = le_Property_Area.transform([request.form["Property_Area"]])
            LoanAmount =  float(request.form["LoanAmount"])
            Loan_Amount_Term =  float(request.form["Loan_Amount_Term"])

            x['ApplicantIncomeLog'] = np.log(ApplicantIncome+1)
            x['LoanAmountLog'] = np.log(LoanAmount+1)
            x['Loan_Amount_Term_Log'] = np.log(Loan_Amount_Term+1)
            
            # prédiction de loan status 
            pred = model.predict(x)
            # décoder le pred [0,1] => ['y','n']
            Loan_Status = le_Loan_Status.inverse_transform(pred)[0]
            # ouvrir la template en utilisant la variable
            return render_template('base.html', pred=Loan_Status)
    # si le template dans un état de GET
    return render_template('base.html')




if __name__ == "__main__" :
    app.run(debug=True)
