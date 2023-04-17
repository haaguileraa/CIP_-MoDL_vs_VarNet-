import os, time
import numpy as np
import torch
import supportingFunctions_torch as sf
import model_torch as mm
from datetime import datetime
from tqdm import tqdm

torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


#--------------------------------------------------------------
# Set these parameters carefully
nLayers = 5
epochs = 50
batchSize = 1
gradientMethod = 'AG'
K = 1
sigma = 0.01
restoreWeights = False

# To train the model with higher K values (K > 1), such as K = 5 or 10, it is better
# to initialize with a pre-trained model with K = 1.
if K > 1:
    restoreWeights = True
    restoreFromModel = '04Jun_0243pm_5L_1K_100E_AG'

if restoreWeights:
    wts = sf.getWeights('savedModels/' + restoreFromModel)

#--------------------------------------------------------------------------
# Generate a meaningful filename to save the trainined models for testing
print('*************************************************')
start_time = time.time()
saveDir = 'savedModels/'
cwd = os.getcwd()
directory = saveDir + datetime.now().strftime("%d%b_%I%M%P_") + \
    str(nLayers) + 'L_' + str(K) + 'K_' + str(epochs) + 'E_' + gradientMethod

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName = directory + '/model'


# save test model

csmT = torch.complex(torch.zeros((None, 12, 256, 232)), torch.zeros((None, 12, 256, 232)))
maskT = torch.complex(torch.zeros((None, 256, 232)), torch.zeros((None, 256, 232)))
atbT = torch.zeros((None, 256, 232, 2))

out = mm.makeModel(atbT, csmT, maskT, False, nLayers, K, gradientMethod)
predTst = out['dc' + str(K)]
predTst = torch.identity(predTst, name='predTst')
sessFileNameTst = directory + '/modelTst'

saver = torch.train.Saver()
with torch.Session() as sess:
    sess.run(torch.global_variables_initializer())
    savedFile = saver.save(sess, sessFileNameTst, latest_filename='checkpointTst')
print('testing model saved:' + savedFile)

# read multi-channel dataset
trnOrg, trnAtb, trnCsm, trnMask = sf.getData('training')
trnOrg, trnAtb = sf.c2r(trnOrg), sf.c2r(trnAtb)



csmP = torch.complex(torch.zeros((None, None, None, None)), torch.zeros((None, None, None, None)))
maskP = torch.complex(torch.zeros((None, None, None)), torch.zeros((None, None, None)))
atbP = torch.zeros((None, None, None, 2))
orgP = torch.zeros((None, None, None, 2))

# creating the dataset
nTrn = trnOrg.shape[0]
nBatch = int(np.floor(np.float32(nTrn) / batchSize))
nSteps = nBatch * epochs

trnData = torch.data.Dataset.from_tensor_slices((orgP, atbP, csmP, maskP))
trnData = trnData.cache()
trnData = trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=1000)
trnData = trnData.batch(batchSize)
trnData = trnData.prefetch(buffer_size=1)
iterator = torch.data.Iterator.from_structure(trnData.output_types, trnData.output_shapes)
orgT, atbT, csmT, maskT = iterator.get_next()

# make training model

out=mm.makeModel(atbT,csmT,maskT,True,nLayers,K,gradientMethod)
predT=out['dc'+str(K)]
predT=torch.identity(predT,name='pred')
loss = torch.reduce_mean(torch.reduce_sum(torch.pow(predT-orgT, 2),axis=0))
torch.summary.scalar('loss', loss)
update_ops = torch.get_collection(torch.GraphKeys.UPDATE_OPS)

with torch.name_scope('optimizer'):
    optimizer = torch.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(torch.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    opToRun=optimizer.apply_gradients(capped_gvs)

# training code

print ('training started at', datetime.now().strftime("%d%b_%I%M%P"))
print ('parameters are: Epochs:',epochs,' BS:',batchSize,'nSteps:',nSteps,'nSamples:',nTrn)

saver = torch.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = torch.placeholder(tf.float32)
lossSumT = torch.summary.scalar("TrnLoss", lossT)

with torch.Session() as sess:
    sess.run(torch.global_variables_initializer())
    if restoreWeights:
        sess=sf.assignWts(sess,nLayers,wts)

    feedDict={orgP:trnOrg,atbP:trnAtb, maskP:trnMask,csmP:trnCsm}
    sess.run(iterator.make_initializer(trnData), feed_dict=feedDict)
    savedFile = saver.save(sess, sessFileName)
    print('Model meta graph saved:' + savedFile)

    writer = torch.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp,_,_=sess.run([loss,opToRun,update_ops])
            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                writer.add_summary(lossSum, ep)
                totalLoss=[]
        except torch.errors.OutOfRangeError:
            break
    savedfile=saver.save(sess, sessFileName, global_step=ep)
    writer.close()

    end_time = time.time()
    print ('total time taken:', ((end_time - start_time) /60))
    print ('training completed at', datetime.now().strftime("%d%b_%I%M%P"))
    print ('*************************************************')

