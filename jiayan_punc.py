# pip install jiayan
# pip install https://github.com/kpu/kenlm/archive/master.zip

from jiayan import load_lm
from jiayan import CRFPunctuator

def punctuate(text):
    lm = load_lm('./jiayan_models/jiayan.klm')
    punctuator = CRFPunctuator(lm, './jiayan_models/cut_model')
    punctuator.load('./jiayan_models/punc_model')
    return punctuator.punctuate(text)

if __name__ == "__main__":
    text = "我爱自然语言处理它是人工智能领域的重要分支"
    punctuated_text = punctuate(text)
    print(punctuated_text)

    text = '天下大乱贤圣不明道德不一天下多得一察焉以自好譬如耳目皆有所明不能相通犹百家众技也皆有所长时有所用虽然不该不遍一之士也判天地之美析万物之理察古人之全寡能备于天地之美称神之容是故内圣外王之道暗而不明郁而不发天下之人各为其所欲焉以自为方悲夫百家往而不反必不合矣后世之学者不幸不见天地之纯古之大体道术将为天下裂'
    punctuated_text = punctuate(text)
    print(punctuated_text)
