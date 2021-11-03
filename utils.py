import re
from typing import Any, Dict


def preprocessor(text: str) -> str:
    """ Remove all HTML Markup tags from string """

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def classify_message(model: Any, message: str) -> Dict[str, Any]:
    """ Classify the message is it Spam or not """

    message = preprocessor(message)
    label = model.predict([message])[0]
    spam_prob = model.predict_proba([message])

    return {'label': label, 'probability': spam_prob[0][1]}

