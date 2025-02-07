import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
from numpy import linspace

def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict_bias_with_attention(model, tokenizer, article):
    inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max_prob, predicted_idx = probabilities[0].max(dim=0)
        return outputs.attentions, probabilities[0], predicted_idx.item()

def get_top_phrases_for_n(tokens, attention, n, top_k=10):
    # Sum all attention weights across all layers and heads to get the total attention score
    summed_attention = torch.stack(attention).sum(dim=0).sum(dim=1).squeeze(0).mean(dim=0)
    n_gram_scores = {}

    for i in range(len(tokens) - n + 1):
        n_gram = ' '.join(tokens[i:i + n])
        score = summed_attention[i:i + n].sum().item()
        n_gram_scores[n_gram] = score

    # Sort phrases with top attention scores
    sorted_n_grams = sorted(n_gram_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_n_grams[:top_k]

def calculate_attention_per_n(tokens, attention, max_n):
    '''Just loops through different n allowed from 1 to max_n and calculates the avg attentions scores of top 3 phrases'''
    summed_attention = torch.stack(attention).sum(dim=0).sum(dim=1).squeeze(0).mean(dim=0)
    avg_scores = []

    for n in range(1, max_n+1):
        top_scores = []

        for i in range(len(tokens) - n + 1):
            score = summed_attention[i:i + n].sum().item()
            top_scores.append(score)

        if top_scores:
            top_scores = sorted(top_scores, reverse=True)[:3]
            avg_scores.append(sum(top_scores) / len(top_scores))
        else:
            avg_scores.append(0)

    return avg_scores


model_path = "article_bias_classifier"
model, tokenizer = load_model(model_path)
# This sample article has bias 'right'
article = "The outpouring , 50 years later , has been nothing short of remarkable .\nThe media are awash in Jack Kennedy tributes , specials , documentaries , books and essays , conjuring up how he lived and how he died . There is an enduring fascination with Camelot , the myth enshrined after his death , and with the myriad theories and counter-theories about Lee Harvey Oswald and that awful day in Dallas .\nBut a partisan battle has also erupted over this question : Was Kennedy really and truly a liberal ?\nWhy , one might ask , does this question still have resonance ? Is it just a way to transpose the politics of 1963 to our 21st-century era of constant political warfare ?\nSure , but it goes deeper than that . Although Kennedy \u2019 s accomplishments were meager in his thousand days , he retains a powerful hold on our imagination . This is in part because he was cut down in his prime , creating a sense of a dream unfulfilled . And though he was a lifelong Democrat , each side wants to claim his legacy .\nBaby boomers , who are forever reliving the sixties , bear part of the blame . But the Kennedy memory obviously has a hold on many who were born well after he died , some of whom want to convert his magic to their cause .\nBoston Globe columnist Jeff Jacoby makes the case for the right :\n\u201c As Democrats maneuvering for the 2016 presidential race , there isn \u2019 t one who would think of disparaging John F. Kennedy \u2019 s stature as a Democratic Party hero . Yet it \u2019 s a pretty safe bet that none would dream of running on Kennedy \u2019 s approach to government or embrace his political beliefs .\n\u201c Today \u2019 s Democratic Party \u2014 the home of Barack Obama , John Kerry , and Al Gore \u2014 wouldn \u2019 t give the time of day to a candidate like JFK .\n\u201c The 35th president was an ardent tax-cutter who championed across-the-board , top-to-bottom reductions in personal and corporate tax rates , slashed tariffs to promote free trade , and even spoke out against the \u2018 confiscatory \u2019 property taxes being levied in too many cities . He was anything but a big-spending , welfare-state liberal . \u2018 I do not believe that Washington should do for the people what they can do for themselves through local and private effort , \u2019 Kennedy bluntly avowed during the 1960 campaign . \u201d\nThat is also the theme of this National Review piece by James Pierson , examining Ira Stoll \u2019 s book \u201c JFK , Conservative \u201d :\n\u201c Now Ira Stoll comes along to make the startling case that JFK was not a liberal at all , but in reality a conservative who ( had he lived ) might have endorsed Ronald Reagan for president and today might be comfortably at home writing editorials for National Review . Most readers will be skeptical of this thesis and are likely to think that the author has taken revisionist history a bit too far . Yet Stoll\u2026makes a strong case that conservatives should stake a claim to President Kennedy as one of their own\u2026\n\u201c JFK appears more conservative to us today than he appeared to his contemporaries because liberalism moved so far to the left in the years after he was killed . \u201d\nThat \u2019 s a telling point , reminding me that Kennedy \u2019 s foe Richard Nixon would be viewed as an outright liberal by today \u2019 s GOP .\nBut historian David Greenberg rejects the argument , making the opposite case in the New Republic :\n\u201c Neither the Camelot mystique nor Kennedy \u2019 s premature death can fully explain his continuing appeal . There was no cult of Warren Harding in 1973 , no William McKinley media blitz in 1951 . I would submit that Kennedy \u2019 s hold on us stems also from the way he used the presidency , his commitment to exercising his power to address social needs , his belief that government could harness expert knowledge to solve problems\u2014in short , from his liberalism .\n\u201c To make that case requires first correcting some misperceptions . Wasn \u2019 t JFK a cold warrior who called on Americans to gird for a \u2018 long twilight struggle \u2019 ? Didn \u2019 t he drag his heels on civil rights ? Didn \u2019 t he give us tax cuts a generation before Ronald Reagan ? While there \u2019 s some truth to those assertions , layers of revisionism and politicized misreadings of Kennedy have come to obscure his true beliefs . During the 1960 presidential campaign , when Republicans tried to make the term liberal anathema , Kennedy embraced it . A liberal , he said in one speech , \u2018 cares about the welfare of the people\u2014their health , their housing , their schools , their jobs , their civil rights , and their civil liberties , \u2019 and under that definition , he said , \u2018 I \u2019 m proud to say I \u2019 m a \u2018 liberal . \u2019\n\u201c Kennedy \u2019 s pledge to \u2018 get America moving again \u2019 should be understood as a part of this collective soul-searching . After the hands-off economic management of President Eisenhower \u2019 s free-marketeers , Kennedy promised an aggressive effort to spur growth and create jobs . After Eisenhower \u2019 s neglect of mounting urban problems , Kennedy promised a federal commitment to education and housing . After Sputnik and the U-2 affair , Kennedy promised a vigorous effort to win hearts and minds around the world . \u201d\nKennedy \u2019 s shortened tenure is such that both sides can cherry-pick his record . Had he lived , would he have battled his party \u2019 s Southern wing and pushed through an LBJ-like civil rights program ? Would he have avoided the quagmire of Vietnam ? We are , because of an assassin \u2019 s bullet , still debating these questions a half century later .\nIt looked for awhile like Jeb Bush might try to follow his brother and father into the White House . Then he made some comments about being out of step with the GOP .\n\u201c New Jersey Gov . Chris Christie is getting all the attention as the flavor of the month for the 2016 Republican presidential nomination . But there is growing chatter in elite New York financial circles that former Florida Gov . Jeb Bush is giving more serious consideration to getting in the race , especially if it appears at any point that Christie is not drawing big national appeal beyond the northeast .\n\u201c Several top GOP sources on Wall Street and in Washington said this week that Bush has moved from almost certainly staying out of the 2016 race to a \u2018 30 percent chance \u2019 of getting in . Several sources mentioned the precise 30 percent odds as up from closer to zero just a few months ago . \u201d\nThese things are getting like weather forecasts , like a 30 percent chance of rain .\nThis first-person account by the Miami Herald \u2019 s Jim Wyss of how he was held captive by Venezuelan authorities for two days is pretty chilling :\n\u201c I was wearing a bulletproof vest , lying flat in the backseat of an unmarked armored car and being escorted by three heavily armed men when I started to worry .\n\u201c At that point I had been in the custody of Venezuela \u2019 s General Directorate of Military Counter Intelligence for 24 hours . I didn \u2019 t know what to expect . All I knew was that a \u2018 commission \u2019 was waiting for me ."
attention_weights, probabilities, predicted_idx = predict_bias_with_attention(model, tokenizer, article)

labels = ['left', 'center', 'right']
print(f'Predicted bias: {labels[predicted_idx]}\nConfidence: {probabilities[predicted_idx].item()*100:.2f}%')

tokens = tokenizer.tokenize(article)
top_phrases_for_print = get_top_phrases_for_n(tokens, attention_weights, n=5, top_k=10)
print("Top 10 influential phrases for n=5:")
for phrase, score in top_phrases_for_print:
    print(f"{phrase}: {score:.1f}")

max_n = 100  # Defines the n interval, from 1 to max_n
n_values = range(1, max_n+1)
avg_attentions = calculate_attention_per_n(tokens, attention_weights, max_n=max_n)

import seaborn as sns
sns.set()
plt.plot(n_values, avg_attentions, lw=2)
plt.xlabel("N (N-Gram Size)", fontsize=14)
plt.ylabel("Average Attention", fontsize=14)
plt.title("Average Attention Score vs. N-Gram Size", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=3, color='grey', linestyle='--', linewidth=2, label='Jump in Attention')
plt.legend()
plt.show()