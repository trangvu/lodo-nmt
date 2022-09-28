# [Domain Generalisation of NMT: Fusing Adapters with Leave-One-Domain-Out Training](https://aclanthology.org/2022.findings-acl.49/)

#### Installation
```shell script
pip install -r requirements.txt

# install fairseq
cd fairseq
pip install --editable ./
```

#### Preprocess
For example, we have data coming from three domains (law, medical, ted) for several language pairs (en-de, sl-it, en
-es, etc). The binarized data should be structure as follow
```
databin
  |--dict.en.txt
  |--dict.de.txt
  |-- one dictionary file for each language
  |--law
     |-- train.en-de.en.bin
     |-- train.en-de.en.idx
     |-- train.en-de.de.bin
     |-- train.en-de.de.idx
     |-- valid.en-de.en.bin
     |-- valid.en-de.en.idx
     |-- valid.en-de.de.bin
     |-- valid.en-de.de.idx
     |-- ... similar for other language pairs
  |--medical
     |-- ... similar as other domains
  |--ted
     |-- ... similar as other domains
```
Note that, there is not necessary to have bilingual data for all language pairs in any given domain.  

Preprocessing and binarization script is similar to standard fairseq NMT. See `preprocess/law/tokenize_and_binarize
.sh` for example. A slightly modified tokenization script `tok.sh` from m2m projects can be found under `m2m_100
` folder.

## Training models
### Train adapter and AdapterFusion
Backbone training scripts can be found in `scripts/run-wmt-backbone.sh` and `scripts/run-md-backbone.sh`.
#### Adapter
Training script: we use `translation_multilingual_multidomain` task
arch `domain_adapter_mbart_large` which defined in the `--user-dir ./mdml-nmt`

An example training script can be found in `scripts/run-adapter.sh`. 
For example, to add target language tag to encoder, we can add `--encoder-langtok tgt`  
```shell script
### example hparam for tag control

./run-adapter.sh exp-name $hparam
```

Decoding script is in `scripts/eval-adapter-fusion-md-backbone.sh` and `scripts/eval-adapter-fusion-wmt-backbone.sh`.
#### Fusion
Training script: we use `translation_multilingual_multidomain` task
arch `domain_adapter_mbart_large` which defined in the `--user-dir ./mdml-nmt`, `adapter-dir` is the path to adapter weights `$ADAPTER_DIR`. The adapter layer for domain `$d` should be saved in `$ADAPTER_DIR/$d`.

An example training script can be found in `runs/run-fusion.sh`. We can control how to form a minibatch to train adapterfusion layer by adding addtional options to the training script.
```shell script
### example hparam for tag control
# LODO with mixed domain batches
hparam="--mixed-batch"

# All domains with homogeneous batches
hparam="--disable-leave-one-out"

# All domains with mixed domain batches
hparam="--disable-leave-one-out --mixed-batch"

./run-fusion.sh exp-name $hparam
```
Decoding script is in `scripts/eval-adapter-fusion-md-backbone.sh` and `scripts/eval-adapter-fusion-wmt-backbone.sh`.

#### NMT with domain discriminators
Joint training NMT objective with domain discrimination objective 
$L_disc = log Pr(d|x) + H(Pr(d|x)$
to learn a domain-aware encoder and domain-agnostic encoder with gradient reversal layer.

Training script: we use `translation_multilingual_multidomain` task, `domain_aware_xent_with_smoothing` criterion,
arch `domain_agnostic_mbart_large` which defined in the `--user-dir ./mdml-nmt`

An example training script can be found in `scripts/run-disc-adv-from-mbart.sh`.
