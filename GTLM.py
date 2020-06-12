__author__ = "Noupin, TensorFlow"

#Third Party Imports
import time
import wandb

#First Party Import
from tunable import Tunable
from preprocessing import Preprocessing
from model import Model

#Initializing project in W&B
#wandb.init(project="lang2lang")

model = Model()

#Portuguese is used as the input language and English is the target language.
print('Started Training.')
for epoch in range(Tunable.tunableVars["EPOCHS"]):
    start = time.time()

    model.train_loss.reset_states()
    model.train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(model.train_dataset):
        model.train_step(inp, tar)

        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, model.train_loss.result(), model.train_accuracy.result()))
            #wandb.log({"accuracy": train_accuracy.result(), "loss": train_loss.result(), "epoch": epoch+1})
        
    if (epoch + 1) % 5 == 0 or epoch + 1 == Tunable.tunableVars["EPOCHS"]:
        ckpt_save_path = model.ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                model.train_loss.result(), 
                                                model.train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
print("Finsihed Training.")


model.translate("este é um problema que temos que resolver.")
print ("Real translation: this is a problem we have to solve .")

model.translate("os meus vizinhos ouviram sobre esta ideia.")
print ("Real translation: and my neighboring homes heard about this idea .")

model.translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.")
print ("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")

#You can pass different layers and attention blocks of the decoder to the plot parameter.
model.translate("este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
print ("Real translation: this is the first book i've ever done.")
