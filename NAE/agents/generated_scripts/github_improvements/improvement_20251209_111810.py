"""
Auto-implemented improvement from GitHub
Source: erma0x/Optimal-Contract-Size-Calculator/app.py
Implemented: 2025-12-09T11:18:10.932107
Usefulness Score: 100
Keywords: def , class , calculate, fit, loss, risk, var, size, stop, loss
"""

# Original source: erma0x/Optimal-Contract-Size-Calculator
# Path: app.py


# Function: __init__
def __init__(self, root):
        self.root = root
        self.root.title("Optimal Contract Size Calculator")
        self.root.configure(bg='#1a1a1a')
        self.root.geometry("600x550")
        
        # Configure style
        self.setup_styles()
        
        # Create GUI elements
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        
        # Configure colors
        bg_color = '#1a1a1a'
        fg_color = '#ffffff'
        entry_bg = '#2d2d2d'
        
        style.configure('Title.TLabel',
                       background=bg_color,
                       foreground='#bb86fc',
                       font=('Arial', 16, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=bg_color,
                       foreground='#03dac6',
                       font=('Arial', 12))
        
        style.configure('TLabel',
                       background=bg_color,
                       foreground=fg_color,
                       font=('Arial', 10))
        
        style.configure('TEntry',
                       fieldbackground=entry_bg,
                       foreground=fg_color,
                       insertcolor=fg_color)
        
        style.configure('TButton',
                       background='#bb86fc',
                       foreground='#000000',
                       font=('Arial', 11, 'bold'),
                       borderwidth=0)
        
        style.map('TButton',
                 background=[('active', '#9965d4')])
        
        style.configure('TRadiobutton',
                       background=bg_color,
                       foreground=fg_color,
                       font=('Arial', 10))
        
    def create_widgets(self):
        # Title
        title = ttk.Label(self.root, text="Optimal Contract Size Calculator", style='Title.TLabel')
        title.pack(pady=20)
        
        # Asset info
        asset_label = ttk.Label(self.root, text=f"Asset Futures: {ASSET}", style='Subtitle.TLabel')
        asset_label.pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(pady=20, padx=40, fill='both', expand=True)
        
        # Trade type selection
        type_frame = tk.Frame(main_frame, bg='#1a1a1a')
        type_frame.pack(pady=10, fill='x')
        
        ttk.Label(type_frame, text="Trade Type:").pack(side='left', padx=5)
        
        self.trade_type = tk.StringVar(value='L')
        ttk.Radiobutton(type_frame, text="Long", variable=self.trade_type, 
                       value='L', style='TRadiobutton').pack(side='left', padx=10)
        ttk.Radiobutton(type_frame, text="Short", variable=self.trade_type, 
                       value='S', style='TRadiobutton').pack(side='left', padx=10)
        
        # Entry price
        self.create_input_row(main_frame, "Entry Price:", "entry")
        
        # Stop Loss
        self.create_input_row(main_frame, "Stop Loss Price:", "sl", color='#cf6679')
        
        # Take Profit
        self.create_input_row(main_frame, "Take Profit Price:", "tp", color='#03dac6')
        
        # Calculate button
        calc_button = ttk.Button(self.root, text="Calculate", command=self.calculate)
        calc_button.pack(pady=20)
        
        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#2d2d2d', relief='solid', borderwidth=1)
        self.results_frame.pack(pady=10, padx=40, fill='both')
        
        self.results_label = tk.Label(self.results_frame, text="", 
                                      bg='#2d2d2d', fg='#ffffff', 
                                      font=('Courier', 10), justify='left')
        self.results_label.pack(pady=15, padx=15)
        
    def create_input_row(self, parent, label_text, entry_name, color='#ffffff'):
        frame = tk.Frame(parent, bg='#1a1a1a')
        frame.pack(pady=8, fill='x')
        
        label = tk.Label(frame, text=label_text, bg='#1a1a1a', 
                        fg=color, font=('Arial', 10), width=20, anchor='w')
        label.pack(side='left', padx=5)
        
        entry = ttk.Entry(frame, width=20)
        entry.pack(side='left', padx=5)
        
        setattr(self, f'{entry_name}_entry', entry)
        
    def calculate_profit_or_loss(self, entry_price, exit_price, trade_type):
        if trade_type == 'L':
            pnl = (exit_price - entry_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        elif trade_type == 'S':
            pnl = (entry_price - exit_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        return pnl
    
    def calculate(self):
        try:
            # Get values
            trade_type = self.trade_type.get()
            entry_price = float(self.entry_entry.get())
            exit_price_sl = float(self.sl_entry.get())
            exit_price_tp = float(self.tp_entry.get())
            
            # Validate inputs
            if trade_type == 'L' and exit_price_sl >= entry_price:
                messagebox.showerror("Error", "Stop Loss must be below Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_sl <= entry_price:
                messagebox.showerror("Error", "Stop Loss must be above Entry Price for Short trades")
                return
            if trade_type == 'L' and exit_price_tp <= entry_price:
                messagebox.showerror("Error", "Take Profit must be above Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_tp >= entry_price:
                messagebox.showerror("Error", "Take Profit must be below Entry Price for Short trades")
                return
            
            # Calculate P&L for 1 contract
            loss_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_sl, trade_type
            )
            
            profit_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_tp, trade_type
            )
            
            # Calculate optimal contracts
            contracts = 1
            while contracts * loss_1_contract_micro > MAX_LOSS_SINGLE_TRADE:
                contracts += 1
                if contracts > 111:
                    messagebox.showerror("Error", "Invalid exit price - risk too high")
                    return
            
            # Calculate ticks
            if trade_type == 'S':
                SL_ticks = (exit_price_sl - entry_price) * 4
                TP_ticks = (entry_price - exit_price_tp) * 4
            elif trade_type == 'L':
                SL_ticks = (entry_price - exit_price_sl) * 4
                TP_ticks = (exit_price_tp - entry_price) * 4
            
            # Format results
            results_text = f"""


# Function: setup_styles
def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        
        # Configure colors
        bg_color = '#1a1a1a'
        fg_color = '#ffffff'
        entry_bg = '#2d2d2d'
        
        style.configure('Title.TLabel',
                       background=bg_color,
                       foreground='#bb86fc',
                       font=('Arial', 16, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=bg_color,
                       foreground='#03dac6',
                       font=('Arial', 12))
        
        style.configure('TLabel',
                       background=bg_color,
                       foreground=fg_color,
                       font=('Arial', 10))
        
        style.configure('TEntry',
                       fieldbackground=entry_bg,
                       foreground=fg_color,
                       insertcolor=fg_color)
        
        style.configure('TButton',
                       background='#bb86fc',
                       foreground='#000000',
                       font=('Arial', 11, 'bold'),
                       borderwidth=0)
        
        style.map('TButton',
                 background=[('active', '#9965d4')])
        
        style.configure('TRadiobutton',
                       background=bg_color,
                       foreground=fg_color,
                       font=('Arial', 10))
        
    def create_widgets(self):
        # Title
        title = ttk.Label(self.root, text="Optimal Contract Size Calculator", style='Title.TLabel')
        title.pack(pady=20)
        
        # Asset info
        asset_label = ttk.Label(self.root, text=f"Asset Futures: {ASSET}", style='Subtitle.TLabel')
        asset_label.pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(pady=20, padx=40, fill='both', expand=True)
        
        # Trade type selection
        type_frame = tk.Frame(main_frame, bg='#1a1a1a')
        type_frame.pack(pady=10, fill='x')
        
        ttk.Label(type_frame, text="Trade Type:").pack(side='left', padx=5)
        
        self.trade_type = tk.StringVar(value='L')
        ttk.Radiobutton(type_frame, text="Long", variable=self.trade_type, 
                       value='L', style='TRadiobutton').pack(side='left', padx=10)
        ttk.Radiobutton(type_frame, text="Short", variable=self.trade_type, 
                       value='S', style='TRadiobutton').pack(side='left', padx=10)
        
        # Entry price
        self.create_input_row(main_frame, "Entry Price:", "entry")
        
        # Stop Loss
        self.create_input_row(main_frame, "Stop Loss Price:", "sl", color='#cf6679')
        
        # Take Profit
        self.create_input_row(main_frame, "Take Profit Price:", "tp", color='#03dac6')
        
        # Calculate button
        calc_button = ttk.Button(self.root, text="Calculate", command=self.calculate)
        calc_button.pack(pady=20)
        
        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#2d2d2d', relief='solid', borderwidth=1)
        self.results_frame.pack(pady=10, padx=40, fill='both')
        
        self.results_label = tk.Label(self.results_frame, text="", 
                                      bg='#2d2d2d', fg='#ffffff', 
                                      font=('Courier', 10), justify='left')
        self.results_label.pack(pady=15, padx=15)
        
    def create_input_row(self, parent, label_text, entry_name, color='#ffffff'):
        frame = tk.Frame(parent, bg='#1a1a1a')
        frame.pack(pady=8, fill='x')
        
        label = tk.Label(frame, text=label_text, bg='#1a1a1a', 
                        fg=color, font=('Arial', 10), width=20, anchor='w')
        label.pack(side='left', padx=5)
        
        entry = ttk.Entry(frame, width=20)
        entry.pack(side='left', padx=5)
        
        setattr(self, f'{entry_name}_entry', entry)
        
    def calculate_profit_or_loss(self, entry_price, exit_price, trade_type):
        if trade_type == 'L':
            pnl = (exit_price - entry_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        elif trade_type == 'S':
            pnl = (entry_price - exit_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        return pnl
    
    def calculate(self):
        try:
            # Get values
            trade_type = self.trade_type.get()
            entry_price = float(self.entry_entry.get())
            exit_price_sl = float(self.sl_entry.get())
            exit_price_tp = float(self.tp_entry.get())
            
            # Validate inputs
            if trade_type == 'L' and exit_price_sl >= entry_price:
                messagebox.showerror("Error", "Stop Loss must be below Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_sl <= entry_price:
                messagebox.showerror("Error", "Stop Loss must be above Entry Price for Short trades")
                return
            if trade_type == 'L' and exit_price_tp <= entry_price:
                messagebox.showerror("Error", "Take Profit must be above Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_tp >= entry_price:
                messagebox.showerror("Error", "Take Profit must be below Entry Price for Short trades")
                return
            
            # Calculate P&L for 1 contract
            loss_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_sl, trade_type
            )
            
            profit_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_tp, trade_type
            )
            
            # Calculate optimal contracts
            contracts = 1
            while contracts * loss_1_contract_micro > MAX_LOSS_SINGLE_TRADE:
                contracts += 1
                if contracts > 111:
                    messagebox.showerror("Error", "Invalid exit price - risk too high")
                    return
            
            # Calculate ticks
            if trade_type == 'S':
                SL_ticks = (exit_price_sl - entry_price) * 4
                TP_ticks = (entry_price - exit_price_tp) * 4
            elif trade_type == 'L':
                SL_ticks = (entry_price - exit_price_sl) * 4
                TP_ticks = (exit_price_tp - entry_price) * 4
            
            # Format results
            results_text = f"""


# Function: create_widgets
def create_widgets(self):
        # Title
        title = ttk.Label(self.root, text="Optimal Contract Size Calculator", style='Title.TLabel')
        title.pack(pady=20)
        
        # Asset info
        asset_label = ttk.Label(self.root, text=f"Asset Futures: {ASSET}", style='Subtitle.TLabel')
        asset_label.pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(pady=20, padx=40, fill='both', expand=True)
        
        # Trade type selection
        type_frame = tk.Frame(main_frame, bg='#1a1a1a')
        type_frame.pack(pady=10, fill='x')
        
        ttk.Label(type_frame, text="Trade Type:").pack(side='left', padx=5)
        
        self.trade_type = tk.StringVar(value='L')
        ttk.Radiobutton(type_frame, text="Long", variable=self.trade_type, 
                       value='L', style='TRadiobutton').pack(side='left', padx=10)
        ttk.Radiobutton(type_frame, text="Short", variable=self.trade_type, 
                       value='S', style='TRadiobutton').pack(side='left', padx=10)
        
        # Entry price
        self.create_input_row(main_frame, "Entry Price:", "entry")
        
        # Stop Loss
        self.create_input_row(main_frame, "Stop Loss Price:", "sl", color='#cf6679')
        
        # Take Profit
        self.create_input_row(main_frame, "Take Profit Price:", "tp", color='#03dac6')
        
        # Calculate button
        calc_button = ttk.Button(self.root, text="Calculate", command=self.calculate)
        calc_button.pack(pady=20)
        
        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#2d2d2d', relief='solid', borderwidth=1)
        self.results_frame.pack(pady=10, padx=40, fill='both')
        
        self.results_label = tk.Label(self.results_frame, text="", 
                                      bg='#2d2d2d', fg='#ffffff', 
                                      font=('Courier', 10), justify='left')
        self.results_label.pack(pady=15, padx=15)
        
    def create_input_row(self, parent, label_text, entry_name, color='#ffffff'):
        frame = tk.Frame(parent, bg='#1a1a1a')
        frame.pack(pady=8, fill='x')
        
        label = tk.Label(frame, text=label_text, bg='#1a1a1a', 
                        fg=color, font=('Arial', 10), width=20, anchor='w')
        label.pack(side='left', padx=5)
        
        entry = ttk.Entry(frame, width=20)
        entry.pack(side='left', padx=5)
        
        setattr(self, f'{entry_name}_entry', entry)
        
    def calculate_profit_or_loss(self, entry_price, exit_price, trade_type):
        if trade_type == 'L':
            pnl = (exit_price - entry_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        elif trade_type == 'S':
            pnl = (entry_price - exit_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        return pnl
    
    def calculate(self):
        try:
            # Get values
            trade_type = self.trade_type.get()
            entry_price = float(self.entry_entry.get())
            exit_price_sl = float(self.sl_entry.get())
            exit_price_tp = float(self.tp_entry.get())
            
            # Validate inputs
            if trade_type == 'L' and exit_price_sl >= entry_price:
                messagebox.showerror("Error", "Stop Loss must be below Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_sl <= entry_price:
                messagebox.showerror("Error", "Stop Loss must be above Entry Price for Short trades")
                return
            if trade_type == 'L' and exit_price_tp <= entry_price:
                messagebox.showerror("Error", "Take Profit must be above Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_tp >= entry_price:
                messagebox.showerror("Error", "Take Profit must be below Entry Price for Short trades")
                return
            
            # Calculate P&L for 1 contract
            loss_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_sl, trade_type
            )
            
            profit_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_tp, trade_type
            )
            
            # Calculate optimal contracts
            contracts = 1
            while contracts * loss_1_contract_micro > MAX_LOSS_SINGLE_TRADE:
                contracts += 1
                if contracts > 111:
                    messagebox.showerror("Error", "Invalid exit price - risk too high")
                    return
            
            # Calculate ticks
            if trade_type == 'S':
                SL_ticks = (exit_price_sl - entry_price) * 4
                TP_ticks = (entry_price - exit_price_tp) * 4
            elif trade_type == 'L':
                SL_ticks = (entry_price - exit_price_sl) * 4
                TP_ticks = (exit_price_tp - entry_price) * 4
            
            # Format results
            results_text = f"""


# Function: create_input_row
def create_input_row(self, parent, label_text, entry_name, color='#ffffff'):
        frame = tk.Frame(parent, bg='#1a1a1a')
        frame.pack(pady=8, fill='x')
        
        label = tk.Label(frame, text=label_text, bg='#1a1a1a', 
                        fg=color, font=('Arial', 10), width=20, anchor='w')
        label.pack(side='left', padx=5)
        
        entry = ttk.Entry(frame, width=20)
        entry.pack(side='left', padx=5)
        
        setattr(self, f'{entry_name}_entry', entry)
        
    def calculate_profit_or_loss(self, entry_price, exit_price, trade_type):
        if trade_type == 'L':
            pnl = (exit_price - entry_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        elif trade_type == 'S':
            pnl = (entry_price - exit_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        return pnl
    
    def calculate(self):
        try:
            # Get values
            trade_type = self.trade_type.get()
            entry_price = float(self.entry_entry.get())
            exit_price_sl = float(self.sl_entry.get())
            exit_price_tp = float(self.tp_entry.get())
            
            # Validate inputs
            if trade_type == 'L' and exit_price_sl >= entry_price:
                messagebox.showerror("Error", "Stop Loss must be below Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_sl <= entry_price:
                messagebox.showerror("Error", "Stop Loss must be above Entry Price for Short trades")
                return
            if trade_type == 'L' and exit_price_tp <= entry_price:
                messagebox.showerror("Error", "Take Profit must be above Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_tp >= entry_price:
                messagebox.showerror("Error", "Take Profit must be below Entry Price for Short trades")
                return
            
            # Calculate P&L for 1 contract
            loss_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_sl, trade_type
            )
            
            profit_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_tp, trade_type
            )
            
            # Calculate optimal contracts
            contracts = 1
            while contracts * loss_1_contract_micro > MAX_LOSS_SINGLE_TRADE:
                contracts += 1
                if contracts > 111:
                    messagebox.showerror("Error", "Invalid exit price - risk too high")
                    return
            
            # Calculate ticks
            if trade_type == 'S':
                SL_ticks = (exit_price_sl - entry_price) * 4
                TP_ticks = (entry_price - exit_price_tp) * 4
            elif trade_type == 'L':
                SL_ticks = (entry_price - exit_price_sl) * 4
                TP_ticks = (exit_price_tp - entry_price) * 4
            
            # Format results
            results_text = f"""


# Function: calculate_profit_or_loss
def calculate_profit_or_loss(self, entry_price, exit_price, trade_type):
        if trade_type == 'L':
            pnl = (exit_price - entry_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        elif trade_type == 'S':
            pnl = (entry_price - exit_price) * TICKS_PER_POINT * US_DOLLAR_PER_TICKS
        return pnl
    
    def calculate(self):
        try:
            # Get values
            trade_type = self.trade_type.get()
            entry_price = float(self.entry_entry.get())
            exit_price_sl = float(self.sl_entry.get())
            exit_price_tp = float(self.tp_entry.get())
            
            # Validate inputs
            if trade_type == 'L' and exit_price_sl >= entry_price:
                messagebox.showerror("Error", "Stop Loss must be below Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_sl <= entry_price:
                messagebox.showerror("Error", "Stop Loss must be above Entry Price for Short trades")
                return
            if trade_type == 'L' and exit_price_tp <= entry_price:
                messagebox.showerror("Error", "Take Profit must be above Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_tp >= entry_price:
                messagebox.showerror("Error", "Take Profit must be below Entry Price for Short trades")
                return
            
            # Calculate P&L for 1 contract
            loss_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_sl, trade_type
            )
            
            profit_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_tp, trade_type
            )
            
            # Calculate optimal contracts
            contracts = 1
            while contracts * loss_1_contract_micro > MAX_LOSS_SINGLE_TRADE:
                contracts += 1
                if contracts > 111:
                    messagebox.showerror("Error", "Invalid exit price - risk too high")
                    return
            
            # Calculate ticks
            if trade_type == 'S':
                SL_ticks = (exit_price_sl - entry_price) * 4
                TP_ticks = (entry_price - exit_price_tp) * 4
            elif trade_type == 'L':
                SL_ticks = (entry_price - exit_price_sl) * 4
                TP_ticks = (exit_price_tp - entry_price) * 4
            
            # Format results
            results_text = f"""


# Function: calculate
def calculate(self):
        try:
            # Get values
            trade_type = self.trade_type.get()
            entry_price = float(self.entry_entry.get())
            exit_price_sl = float(self.sl_entry.get())
            exit_price_tp = float(self.tp_entry.get())
            
            # Validate inputs
            if trade_type == 'L' and exit_price_sl >= entry_price:
                messagebox.showerror("Error", "Stop Loss must be below Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_sl <= entry_price:
                messagebox.showerror("Error", "Stop Loss must be above Entry Price for Short trades")
                return
            if trade_type == 'L' and exit_price_tp <= entry_price:
                messagebox.showerror("Error", "Take Profit must be above Entry Price for Long trades")
                return
            if trade_type == 'S' and exit_price_tp >= entry_price:
                messagebox.showerror("Error", "Take Profit must be below Entry Price for Short trades")
                return
            
            # Calculate P&L for 1 contract
            loss_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_sl, trade_type
            )
            
            profit_1_contract_micro = self.calculate_profit_or_loss(
                entry_price, exit_price_tp, trade_type
            )
            
            # Calculate optimal contracts
            contracts = 1
            while contracts * loss_1_contract_micro > MAX_LOSS_SINGLE_TRADE:
                contracts += 1
                if contracts > 111:
                    messagebox.showerror("Error", "Invalid exit price - risk too high")
                    return
            
            # Calculate ticks
            if trade_type == 'S':
                SL_ticks = (exit_price_sl - entry_price) * 4
                TP_ticks = (entry_price - exit_price_tp) * 4
            elif trade_type == 'L':
                SL_ticks = (entry_price - exit_price_sl) * 4
                TP_ticks = (exit_price_tp - entry_price) * 4
            
            # Format results
            results_text = f"""

