import argparse
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
import utils
import CCB_model
from vqa_debias_loss_functions import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch.nn.functional as F
import torch._utils
import base_model
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', default=False,
        help="Cache image features in RAM. Makes things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--dataset', default='cpv2', help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '--debias', default="CCB_loss",
        choices=["learned_mixin", "reweight", "bias_product", "none", "CCB_loss"],
        help="Kind of ensemble loss to use")
    # Arguments from the original model, we leave this default, except we
    # set --epochs to 15 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='CCB_model', choices=["baseline0_newatt", "CCB_model"])  # update 190
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--prob', type=str, default="sigmoid", choices=['softmax','sigmoid'])
    parser.add_argument('--model_state', type=str, default='logs/exp_cc_none_0.42/57.99.pth')
    args = parser.parse_args()
    return args

def get_v_q_feature(img_id, question, dictionary):
    fe = torch.load('data/rcnn_feature/' + str(img_id) + '.pth')['image_feature']
    tokens = dictionary.tokenize(question, False)
    tokens = tokens[:14]
    if len(tokens) < 14:
        # Note here we pad in front of the sentence
        padding = [dictionary.padding_idx] * (14 - len(tokens))
        tokens = padding + tokens
    utils.assert_eq(len(tokens), 14)
    qe = torch.from_numpy(np.array(tokens))
    return fe, qe

def Visualize(image_id,question,Answer,pred,att,label_dictionary):
    image_address = "val2014/COCO_val2014_0000000" + str(image_id) + ".jpg"
    bbox = torch.load('data/rcnn_feature/' + str(image_id) + '.pth')['spatial_feature']
    logits = {}

    logits['answer'] = pred['answer'].data.cpu().tolist()[0]
    answer = np.array(logits['answer'])
    answer_top10 = answer.argsort()[::-1][0:5]

    answer_p = []
    answer_word = []

    for i in answer_top10:
        answer_p.append(answer[i])
        answer_word.append(label_dictionary[i])

    print('answer:', answer_word)
    print('answer probability:', answer_p)

    if pred.has_key('fq'):
        logits['content'] = pred['fq'].data.cpu().tolist()[0]
        logits['context'] = pred['vq'].data.cpu().tolist()[0]
        content = np.array(logits['content'])
        content_top10 = content.argsort()[::-1][0:5]
        context = np.array(logits['context'])
        context_top10 = context.argsort()[::-1][0:5]

        content_p = []
        content_word = []
        context_p = []
        context_word = []

        for i in content_top10:
            content_p.append(content[i])
            content_word.append(label_dictionary[i])
        for i in context_top10:
            context_p.append(context[i])
            context_word.append(label_dictionary[i])

        print('content:', content_word)
        print('content probability:', content_p)
        print('context:', context_word)
        print('context probability:', context_p)

    # visualize image question attention

    '''
    grid = plt.GridSpec(2, 3, wspace=0.7, hspace=0.3)
    plt.subplot(grid[0,0:3])
    image = mpimg.imread(image_address)
    plt.imshow(image)

    a_score = att.data.cpu().tolist()
    a_score_np = np.array(a_score[0])
    a_score_np = np.round(a_score_np, 2)
    a_score_np = a_score_np.flatten()
    a_top3 = a_score_np.argsort()[::-1][0:1]

    for j in a_top3:
        rect = plt.Rectangle((bbox[j][0], bbox[j][1]), bbox[j][2] - bbox[j][0], bbox[j][3] - bbox[j][1], fill=False,
                             edgecolor='lime', linewidth=2)
        plt.gca().add_patch(rect)
        plt.gca().text(bbox[j][0], bbox[j][1], a_score_np[j], size='xx-large', color='blue',
                       ) #bbox={'facecolor': 'blue', 'alpha': 0.5}
    plt.axis('off')
    plt.title(question)
    
    
    '''
    '''
    plt.subplot(grid[1,0])
    plt.tick_params(labelsize=25)
    plt.barh(content_word, content_p,facecolor='gold',height=0.5,edgecolor='black',alpha=1)
    plt.title('Content')
    plt.xticks([])

    plt.subplot(grid[1,1])
    plt.tick_params(labelsize=25)
    plt.barh(context_word, context_p,facecolor='gold',height=0.5,edgecolor='black',alpha=1)
    plt.title('Context')
    plt.xticks([])

    plt.subplot(grid[1,2])
    plt.tick_params(labelsize=25)
    plt.barh(answer_word, answer_p,facecolor='gold',height=0.5,edgecolor='black',alpha=1)
    plt.title(Answer)
    plt.xticks([])
    
    '''
    plt.tick_params(labelsize=25)
    plt.barh(answer_word, answer_p, facecolor='gold', height=0.5, edgecolor='black', alpha=1)
    plt.title(Answer)
    plt.xticks([])

    #save
    '''
    plt.savefig("visualize_attention.png", dpi=600)
    plt.clf()
    plt.barh(content_word, content_p)
    plt.savefig("content_p.png", dpi=600)
    plt.clf()
    plt.barh(context_word, context_p)
    plt.savefig("context.png", dpi=600)
    plt.clf()
    plt.barh(answer_word, answer_p)
    plt.savefig("answer_p.png", dpi=600)
    '''

    plt.show()

def main():
    args = parse_args()
    dataset = args.dataset

    if dataset == 'cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset == 'cpv2' or dataset == 'v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=args.cache_features)
    lable2answer=eval_dset.label2ans

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(CCB_model, constructor)(eval_dset, args.num_hid).cuda()  # base_model, CC_model
    # model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()

    if args.debias == "bias_product":
        model.debias_loss_fn = BiasProduct()
    elif args.debias == "none":
        model.debias_loss_fn = Plain()
    elif args.debias == "reweight":
        model.debias_loss_fn = ReweightByInvBias()
    elif args.debias == "learned_mixin":
        model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    elif args.debias == "CCB_loss":
        model.debias_loss_fn = CCB_loss(args.entropy_penalty)

    else:
        raise RuntimeError(args.mode)

    model_state = torch.load(args.model_state)
    model.load_state_dict(model_state)
    model = model.cuda()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print("Starting eval...")

    image_id = 507533
    question = "What color are the bananas?"
    Answer=''
    """
    image_id=18896   
    question="How many feet do the women on the ground?"
    image_id=389753
    question="What color is the counter?"
    image_id=507533 
    question="What color are the bananas?"
    """

    v, q = get_v_q_feature(image_id, question, dictionary)
    v = torch.Tensor(v)
    v = v.unsqueeze(0)
    q = q.unsqueeze(0)
    v = Variable(v, requires_grad=False).cuda()
    q = Variable(q, requires_grad=False).cuda()
    pred, _, _, att = model(v, q, None, None, None)

    if args.prob=='softmax':
        pred['answer'] = F.softmax(pred['answer'])
        if args.model == "CCB_model":
            pred['fq'] = F.softmax(pred['fq'])
            pred['vq'] = F.softmax(pred['vq'])
    else:
        pred['answer'] = F.sigmoid(pred['answer'])
        if args.model == "CCB_model":
            pred['fq'] = F.sigmoid(pred['fq'])
            pred['vq'] = F.sigmoid(pred['vq'])




    Visualize(image_id,question,Answer,pred,att,lable2answer)

if __name__ == '__main__':
    main()