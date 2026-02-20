"""
Auto-implemented improvement from GitHub
Source: urshayan/KellyOne/app.py
Implemented: 2025-12-09T11:17:16.851366
Usefulness Score: 80
Keywords: def , risk
"""

# Original source: urshayan/KellyOne
# Path: app.py


# Function: calc_kelly
def calc_kelly():
    ## All of the Logic Goes Here!
    try:
        p = float(p_value.get())
        b = float(b_value.get())
        cap = float(capital.get())

        if not (0 < p < 1):
            messagebox.showerror("Invalid Input","Probability must be between 0 & 1")
            return
    
        q = 1 - p
        ### Calculation goes here!
        f_star = (b * p - q) / b 

        if f_star <= 0:
            result.config(text="Kelly Suggests not taking this bet", fg="red")
            suggestion.config(text="DO NOT TAKE THIS TRADE!", fg="red")
        else:
            invest_amount = f_star * cap 
            result.config(text=f"Optimal Fraction : {f_star: .2%} \n Recommended Investment : {invest_amount: .2f} " , fg="green")
            half_kely = round(invest_amount/2 , 2)
            quarter_kely = round(invest_amount/4 , 2) 
            
            suggestion.config(text=f" Suggested \n  Half Kelly Investment Amount: {half_kely} \n Quarter Kelly Investment Amount : {quarter_kely} " , fg="blue")
    except ValueError:
        messagebox.showerror("Error!","Please Enter Valid Numbers!")

        

