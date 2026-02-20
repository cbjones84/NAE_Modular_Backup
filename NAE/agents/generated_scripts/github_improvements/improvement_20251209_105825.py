"""
Auto-implemented improvement from GitHub
Source: Marsan-Ma-zz/tf_chatbot_seq2seq_antilm/app.py
Implemented: 2025-12-09T10:58:25.967279
Usefulness Score: 100
Keywords: def , class , tensorflow, model, predict, size
"""

# Original source: Marsan-Ma-zz/tf_chatbot_seq2seq_antilm
# Path: app.py


# Function: __init__
def __init__(self, args, debug=False):
        start_time = datetime.now()

        # flow ctrl
        self.args = args
        self.debug = debug
        self.fbm_processed = []
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_usage)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # Create model and load parameters.
        self.args.batch_size = 1  # We decode one sentence at a time.
        self.model = create_model(self.sess, self.args)

        # Load vocabularies.
        self.vocab_path = os.path.join(self.args.data_dir, "vocab%d.in" % self.args.vocab_size)
        self.vocab, self.rev_vocab = data_utils.initialize_vocabulary(self.vocab_path)
        print("[ChatBot] model initialize, cost %i secs" % (datetime.now() - start_time).seconds)

        # load yaml setup
        self.FBM_API = "https://graph.facebook.com/v2.6/me/messages"
        with open("config.yaml", 'rt') as stream:
            try:
                cfg = yaml.load(stream)
                self.FACEBOOK_TOKEN = cfg.get('FACEBOOK_TOKEN')
                self.VERIFY_TOKEN = cfg.get('VERIFY_TOKEN')
            except yaml.YAMLError as exc:
                print(exc)


    def process_fbm(self, payload):
        for sender, msg in self.fbm_events(payload):
            self.fbm_api({"recipient": {"id": sender}, "sender_action": 'typing_on'})
            resp = self.gen_response(msg)
            self.fbm_api({"recipient": {"id": sender}, "message": {"text": resp}})
            if self.debug: print("%s: %s => resp: %s" % (sender, msg, resp))
            

    def gen_response(self, sent, max_cand=100):
        sent = " ".join([w.lower() for w in jieba.cut(sent) if w not in [' ']])
        # if self.debug: return sent
        raw = get_predicted_sentence(self.args, sent, self.vocab, self.rev_vocab, self.model, self.sess, debug=False)
        # find bests candidates
        cands = sorted(raw, key=lambda v: v['prob'], reverse=True)[:max_cand]
        
        if max_cand == -1:  # return all cands for debug
            cands = [(r['prob'], ' '.join([w for w in r['dec_inp'].split() if w[0] != '_'])) for r in cands]
            return cands
        else:
            cands = [[w for w in r['dec_inp'].split() if w[0] != '_'] for r in cands]
            return ' '.join(choice(cands)) or 'No comment'


    def gen_response_debug(self, sent, args=None):
        sent = " ".join([w.lower() for w in jieba.cut(sent) if w not in [' ']])
        raw = get_predicted_sentence(args, sent, self.vocab, self.rev_vocab, self.model, self.sess, debug=False, return_raw=True)
        return raw


    #------------------------------
    #   FB Messenger API
    #------------------------------
    def fbm_events(self, payload):
        data = json.loads(payload.decode('utf8'))
        if self.debug: print("[fbm_payload]", data)
        for event in data["entry"][0]["messaging"]:
            if "message" in event and "text" in event["message"]:
                q = (event["sender"]["id"], event["message"]["seq"])
                if q in self.fbm_processed:
                    continue
                else:
                    self.fbm_processed.append(q)
                    yield event["sender"]["id"], event["message"]["text"]


    def fbm_api(self, data):
        r = requests.post(self.FBM_API,
            params={"access_token": self.FACEBOOK_TOKEN},
            data=json.dumps(data),
            headers={'Content-type': 'application/json'})
        if r.status_code != requests.codes.ok:
            print("fb error:", r.text)
        if self.debug: print("fbm_send", r.status_code, r.text)
        

#---------------------------
#   Server
#---------------------------

