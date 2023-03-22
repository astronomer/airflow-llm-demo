Following are the steps to manually build the fake customer tweet dataset.

```bash
curl -o unprocessed.tar.gz https://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz
tar -xvf unprocessed.tar.gz sorted_data/toys_\&_games/all.review sorted_data/toys_\&_games/negative.review sorted_data/toys_\&_games/positive.review
grep -v --color='auto' -P '[^\x00-\x7F]'  sorted_data/toys_\&_games/all.review | awk '/\<review_text\>/,/\<\/review_text\>/' | sed 's/<review_text>/---/g' | sed 's/<\/review_text>//g' > all.txt
awk '/---/ {printf "\n%s\n",$0;next} {printf "%s ",$0}' all.txt | sed 's/---//g' | sed '/^$/d' > all.txt1
grep -v --color='auto' -P '[^\x00-\x7F]'  sorted_data/toys_\&_games/negative.review | awk '/\<review_text\>/,/\<\/review_text\>/' | sed 's/<review_text>/---/g' | sed 's/<\/review_text>//g' > negative.txt
awk '/---/ {printf "\n%s\n",$0;next} {printf "%s ",$0}' negative.txt | sed 's/---//g' | sed '/^$/d' > negative.txt1
grep -v --color='auto' -P '[^\x00-\x7F]'  sorted_data/toys_\&_games/positive.review | awk '/\<review_text\>/,/\<\/review_text\>/' | sed 's/<review_text>/---/g' | sed 's/<\/review_text>//g' > positive.txt
awk '/---/ {printf "\n%s\n",$0;next} {printf "%s ",$0}' positive.txt | sed 's/---//g' | sed '/^$/d' > positive.txt1
curl -o FJAZ_Sa.wav http://www2.imm.dtu.dk/~lfen/elsdsr/sounds/FJAZ_Sa.wav
curl -o FUAN_Sa.wav http://www2.imm.dtu.dk/~lfen/elsdsr/sounds/FUAN_Sa.wav
curl -o MASM_Sg.wav http://www2.imm.dtu.dk/~lfen/elsdsr/sounds/MASM_Sg.wav
curl -o MKBP_Sg.wav http://www2.imm.dtu.dk/~lfen/elsdsr/sounds/MKBP_Sg.wav
curl -o FDHH_Sr26.wav http://www2.imm.dtu.dk/~lfen/elsdsr/sounds/FDHH_Sr26.wav
curl -o MLKH_Sr37.wav http://www2.imm.dtu.dk/~lfen/elsdsr/sounds/MLKH_Sr37.wav
```
  
```python
import pandas as pd
with open('include/data/negative.txt1') as f:
    negdf = pd.DataFrame(f.readlines(), columns=['review_text']).drop_duplicates().reset_index(drop=True)
    negdf = negdf.replace(r'^\s*$', np.nan, regex=True).dropna()
    negdf['review_text'] = negdf['review_text'].apply(lambda x: x.replace("\n",""))
    negdf['label']=0
with open('include/data/positive.txt1') as f:
    posdf = pd.DataFrame(f.readlines(), columns=['review_text']).drop_duplicates().reset_index(drop=True)
    posdf = posdf.replace(r'^\s*$', np.nan, regex=True).dropna()
    posdf['review_text'] = posdf['review_text'].apply(lambda x: x.replace("\n",""))
    posdf['label']=1
df = pd.concat([posdf, negdf]).to_parquet('include/data/comment_training.parquet')
df = pd.read_parquet('include/data/comment_training.parquet')
df.loc[df.duplicated()]
df.isna().values.any()


with open('include/data/all.txt1') as f:
    alldf = pd.DataFrame(f.readlines(), columns=['review_text']).drop_duplicates().reset_index(drop=True)
    alldf = alldf.replace(r'^\s*$', np.nan, regex=True).dropna()
    alldf['review_text'] = alldf['review_text'].apply(lambda x: x.replace("\n",""))
    alldf=alldf.drop_duplicates().reset_index(drop=True)
ordersdf = pd.read_csv('include/data/orders.csv')[['customer_id','order_date']].sample(len(alldf), axis=0, replace=True).reset_index(drop=True)
df = pd.concat([ordersdf, alldf], axis=1)
df.rename(columns={'order_date': 'date'}).to_parquet('include/data/twitter_comments.parquet')
df = pd.read_parquet('include/data/twitter_comments.parquet')
df.loc[df.duplicated()]
len(alldf) == len(ordersdf)
df.isna().values.any()
```