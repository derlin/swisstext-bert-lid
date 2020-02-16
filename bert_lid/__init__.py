import torch
import pytorch_pretrained_bert as bert
import os
import re
import logging

_DEFAULT_MODEL_DIR = os.path.realpath(os.path.join(
    os.path.dirname(__file__), 'models', 'default'))

logger = logging.getLogger(__file__)


class BertLid:

    def __init__(self, model_dir=_DEFAULT_MODEL_DIR, device=None, chunk_size=2000):
        """
        :param model_dir: root directory a 'bert' directory with the model and an 'environ.txt'
        :param device: either a string ('cuda', 'cpu', ...) or a torch.device
        :param chunk_size: max batch_size during predict (depending on your CPU/GPU capacity)
        """

        self.chunk_size = chunk_size

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Config from training
        self.config = torch.load(os.path.join(model_dir, 'bert', 'training_args.bin'))
        self.max_seq_length = getattr(self.config, 'max_seq_length', 128)
        logging.info(f'MAX_SEQ_LENGTH {self.max_seq_length}')

        # Classification labels and class weights
        self.labels, weights = [], []
        with open(os.path.join(model_dir, 'environ.txt')) as f:
            lines = [l.strip() for l in f]
            self.labels = re.search('="(.*)"', lines[0]).group(1).split(',')
            logging.info(f'LABELS {self.labels}')
            if len(lines) > 1 and '=' in lines[1]:
                weights = [float(x) for x in re.search('="(.*)"', lines[1]).group(1).split(',')]
                logging.info(f'WEIGHTS {weight}')

        # BERT
        self.tokenizer = bert.BertTokenizer.from_pretrained(
            self.config.bert_model, do_lower_case=self.config.do_lower_case, do_basic_tokenize=True)

        self.model = bert.BertForSequenceClassification.from_pretrained(
            os.path.join(model_dir, 'bert'),
            from_tf=False,
            num_labels=len(self.labels)
        ).to(self.device)

        self.model.eval()  # HIGHLY IMPORTANT !!!

    # == Data preprocessing functions
    def _preproc_single_sentence(self, s):
        # see https://github.com/google-research/bert/blob/master/run_classifier.py
        segment_ids = [0] * self.max_seq_length

        tokens = self.tokenizer.tokenize(s)[:self.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])

        input_size = len(input_ids)
        padding_size = (self.max_seq_length - input_size)

        input_mask = ([1] * input_size) + ([0] * padding_size)
        input_ids += [0] * padding_size

        assert all(len(x) == self.max_seq_length for x in [input_ids, segment_ids, input_mask])
        return input_ids, segment_ids, input_mask, 0

    def create_features(self, sentences):
        return [torch.tensor(x).to(self.device) for x in zip(*map(self._preproc_single_sentence, sentences))]

    # == Predictions functions

    def _predict(self, sentences):
        for i in range(0, len(sentences), self.chunk_size):
            input_ids, segment_ids, input_mask, _ = self.create_features(sentences[i:i + self.chunk_size])
            with torch.no_grad():
                logits = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                yield torch.softmax(logits, dim=1)

    def predict_lang(self, sentences, mode='col'):
        """
        Predict the language of sentences.

        :param sentences: a list of sentences to evaluate
        :param mode: how to return the results. If 'col', returns two lists: ([label], [proba]).
            If 'row', returns a list of triples [(sentence, label, proba)]
        :return: depends on the mode parameter. Note: the probabilities are between 0 and 1.
        """
        if mode not in ['row', 'col']:
            raise Exception('Wrong mode received', mode)

        probas, labels = [], []
        for out in self._predict(sentences):
            best_probas, best_idx = torch.max(out, dim=1)
            probas += best_probas.tolist()
            labels += [self.labels[x] for x in best_idx.tolist()]

        if mode == 'row':
            return list(zip(sentences, labels, probas))
        elif mode == 'col':
            return labels, probas

    def predict_label(self, sentences, label='gsw'):
        """
        Return the probability of sentences to be in a given language.
        :param sentences: a list of sentences to evaluate
        :param label: the target language
        :return: a list of probabilities (0 <= p <= 1)
        """
        if label not in self.labels:
            raise Exception(f'{label} is unknown to this model.')
        label_idx = self.labels.index(label)

        probas = []
        for out in self._predict(sentences):
            probas += out[:, label_idx].tolist()
        return probas

    def predict(self, *args, **kwargs):
        return self.predict_label(*args, **kwargs)
