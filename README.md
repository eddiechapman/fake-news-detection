# Fake news detection

Perform stylometric feature selection on a fake news dataset.

## Installation

```
git clone https://github.com/eddiechapman/fake-news-detection.git
cd fake-news-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the following command:

```
python3 features.py
```

Data will be read from `/data/test.csv` and `/data/train.csv`


Two CSV files will be created:

- `data/test_features.csv`
- `data/train_features.csv`

Four PNG figures will be created:

- `figs/author_features.png`
- `figs/content_features.png`
- `figs/sentiment.png`
- `figs/unique_words.png`

## Resources

[Email spam detection tutorial](https://github.com/patrycas/How-to-identify-Spam-using-Natural-Language-Processing-NLP-/blob/main/spam_detection.ipynb)

[Text classification tutorial](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/)


[Stylometric Text Classification](https://github.com/Hassaan-Elahi/Writing-Styles-Classification-Using-Stylometric-Analysis/blob/master/Code/main.py)