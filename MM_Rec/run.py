import numpy as np
import torch
import json
import logging
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path
import utils
import os
from dataloader import DataLoaderTrain, DataLoaderTest
from torch.utils.data import Dataset, DataLoader
from preprocess import read_news_bert, get_doc_input_bert,read_news_image_size,read_news_image
from model import mmrec
from parameters import parse_args
from transformers import AutoTokenizer
from model import BertConfig
from metrics import ndcg_score, mrr_score, hit_at_k

def build_finetuneset(config):
    """원 MM-Rec: 텍스트 마지막 4층 + visual/co-attention 전부 + user encoder."""
    n_t = int(getattr(config, "num_hidden_layers", 8))
    v_n = int(getattr(config, "v_num_hidden_layers", 2))
    c_n = len(getattr(config, "v_biattention_id", [0, 1]))
    names = {
        "news_encoder.bert.t_pooler",
        "news_encoder.bert.v_pooler",
        "news_encoder.bert.t_pooler_new",
        "news_encoder.bert.v_pooler_new",
        "news_encoder.bert.v_embeddings",
        "user_encoder",
    }
    for i in range(max(0, n_t - 4), n_t):
        names.add(f"news_encoder.bert.encoder.layer.{i}")
    for i in range(v_n):
        names.add(f"news_encoder.bert.encoder.v_layer.{i}")
    for i in range(c_n):
        names.add(f"news_encoder.bert.encoder.c_layer.{i}")
    return names


def _param_trainable(name, prefixes):
    for p in prefixes:
        if name == p or name.startswith(p + "."):
            return True
    return False


def _news_tsv_path(args):
    if os.path.isabs(args.news_file):
        return args.news_file
    return os.path.join(args.root_data_dir, args.news_file)


def _map_news_to_image(news_index, news_imageid_dict):
    news_id2image_id = {}
    for news_id, image_id in news_imageid_dict.items():
        if news_id in news_index:
            news_id2image_id[news_index[news_id]] = image_id
    return news_id2image_id


def _reduce_mean(tensor, enable_hvd, hvd):
    if enable_hvd:
        return hvd.allreduce(tensor)
    return tensor


def _selection_key(name):
    name = (name or "MRR").strip()
    if name.lower() in {"ndcg@5", "ndcg5"}:
        return "NDCG@5"
    return "MRR"


def _device(args):
    if args.enable_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NewsDataset(Dataset):
    def __init__(self, data, roi_feature, roi_location, roi_mask, image_whole, news_id2image_id):
        self.data = data
        self.roi_feature = roi_feature
        self.roi_location = roi_location
        self.roi_mask = roi_mask
        self.image_whole = image_whole
        self.news_id2image_id = news_id2image_id

    def __getitem__(self, idx):
        image_idx = self.news_id2image_id[idx] if idx in self.news_id2image_id else 0
        return (
            self.data[idx],
            self.roi_feature[image_idx],
            self.roi_location[image_idx],
            self.roi_mask[image_idx],
            self.image_whole[image_idx],
        )

    def __len__(self):
        return self.data.shape[0]


def news_collate_fn(arr):
    mini_batch_news_feature = []
    mini_batch_news_roi_feature = []
    mini_batch_news_roi_location = []
    mini_batch_news_roi_mask = []
    mini_batch_news_image_whole = []
    for news_feature, roi_feature, roi_location, roi_mask, image_whole in arr:
        mini_batch_news_feature.append(news_feature)
        mini_batch_news_roi_feature.append(roi_feature)
        mini_batch_news_roi_location.append(roi_location)
        mini_batch_news_roi_mask.append(roi_mask)
        mini_batch_news_image_whole.append(image_whole)
    mini_batch_news_feature = np.array(mini_batch_news_feature)
    mini_batch_news_roi_feature = np.array(mini_batch_news_roi_feature)
    mini_batch_news_roi_location = np.array(mini_batch_news_roi_location)
    mini_batch_news_roi_mask = np.array(mini_batch_news_roi_mask)
    mini_batch_news_image_whole = np.array(mini_batch_news_image_whole)

    batch_size = mini_batch_news_roi_location.shape[0]
    input_imgs = np.concatenate([mini_batch_news_image_whole, mini_batch_news_roi_feature], 1)
    image_loc = np.concatenate(
        [np.expand_dims(np.array([[0, 0, 1, 1, 1]], dtype=np.float32).repeat(batch_size, 0), 1),
         mini_batch_news_roi_location],
        1,
    )
    image_mask = np.concatenate(
        [np.array([[1]], dtype=np.float32).repeat(batch_size, 0), mini_batch_news_roi_mask], 1
    )
    return (
        torch.LongTensor(mini_batch_news_feature),
        torch.FloatTensor(input_imgs),
        torch.FloatTensor(image_loc),
        torch.FloatTensor(image_mask),
    )


def encode_news(
    args,
    model,
    news_combined,
    news_roi_feature,
    news_roi_location,
    news_roi_mask,
    news_image_whole,
    news_id2image_id,
):
    news_dataset = NewsDataset(
        news_combined,
        news_roi_feature,
        news_roi_location,
        news_roi_mask,
        news_image_whole,
        news_id2image_id,
    )
    news_dataloader = DataLoader(
        news_dataset,
        batch_size=args.batch_size * 4,
        num_workers=0,
        collate_fn=news_collate_fn,
    )
    news_scoring_t = []
    news_scoring_v = []
    device = _device(args)
    model.eval()
    with torch.no_grad():
        for input_ids, input_imgs, image_loc, image_mask in tqdm(news_dataloader, desc="encode_news"):
            input_ids = input_ids.to(device)
            feature_dict = {
                "input_txt": torch.narrow(input_ids, 1, 0, args.num_words_title),
                "input_imgs": input_imgs.to(device),
                "image_loc": image_loc.to(device),
                "token_type_ids": torch.narrow(
                    input_ids, 1, args.num_words_title, args.num_words_title
                ),
                "attention_mask": torch.narrow(
                    input_ids, 1, args.num_words_title * 2, args.num_words_title
                ),
                "image_attention_mask": image_mask.to(device),
            }
            news_vec_t, news_vec_v = model.news_encoder(**feature_dict)
            news_scoring_t.extend(news_vec_t.detach().cpu().numpy())
            news_scoring_v.extend(news_vec_v.detach().cpu().numpy())
    news_scoring_t = np.array(news_scoring_t)
    news_scoring_v = np.array(news_scoring_v)
    logging.info("news scoring num: {}".format(news_scoring_t.shape[0]))
    return news_scoring_t, news_scoring_v


def score_impressions(
    args,
    model,
    news_index,
    news_scoring_t,
    news_scoring_v,
    news_id2image_id,
    data_dir,
    hvd_size,
    hvd_rank,
    hvd_local_rank,
    result_name=None,
):
    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring_t=news_scoring_t,
        news_scoring_v=news_scoring_v,
        word_dict=None,
        news_bias_scoring=None,
        data_dir=data_dir,
        filename_pat="test_*.tsv",
        args=args,
        world_size=hvd_size,
        news_id2image_id=news_id2image_id,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=getattr(args, "enable_prefetch", False),
        enable_shuffle=False,
        enable_gpu=args.enable_gpu,
    )
    MRR, nDCG5, HIT1 = [], [], []
    device = _device(args)
    model.eval()
    with torch.no_grad():
        for cnt, (log_vecs_t, log_vecs_v, log_masks, news_vecs_t, news_vecs_v, news_bias, labels) in enumerate(
            dataloader
        ):
            for user_vec_t, user_vec_v, news_vec_t, news_vec_v, label, log_mask in zip(
                log_vecs_t, log_vecs_v, news_vecs_t, news_vecs_v, labels, log_masks
            ):
                if label.mean() == 0 or label.mean() == 1:
                    continue
                user_vec_t = torch.as_tensor(user_vec_t, dtype=torch.float32, device=device).unsqueeze(0)
                user_vec_v = torch.as_tensor(user_vec_v, dtype=torch.float32, device=device).unsqueeze(0)
                news_vec_t = torch.as_tensor(news_vec_t, dtype=torch.float32, device=device).unsqueeze(0)
                news_vec_v = torch.as_tensor(news_vec_v, dtype=torch.float32, device=device).unsqueeze(0)
                log_mask = torch.as_tensor(log_mask, dtype=torch.float32, device=device).unsqueeze(0)
                user_vecs = model.user_encoder(
                    news_vec_t, news_vec_v, user_vec_t, user_vec_v, log_mask
                )
                score = torch.sum((news_vec_t + news_vec_v) * user_vecs, -1)
                score = score.squeeze(0).cpu().detach().numpy()
                MRR.append(mrr_score(label, score))
                nDCG5.append(ndcg_score(label, score, k=5))
                HIT1.append(hit_at_k(label, score, k=1))
            if cnt % args.log_steps == 0 and MRR:
                logging.info(
                    "[{}] Ed: {}: MRR={:.2f}\tNDCG@5={:.2f}\tHit@1={:.2f}".format(
                        hvd_rank,
                        cnt * args.batch_size,
                        np.mean(MRR) * 100,
                        np.mean(nDCG5) * 100,
                        np.mean(HIT1) * 100,
                    )
                )

    n = len(MRR)
    metrics = {
        "MRR": float(np.mean(MRR)) if n else 0.0,
        "NDCG@5": float(np.mean(nDCG5)) if n else 0.0,
        "Hit@1": float(np.mean(HIT1)) if n else 0.0,
        "n": n,
    }
    if result_name:
        with open(os.path.join(args.log_dir, f"{result_name}_{hvd_rank}.txt"), "w") as fout:
            fout.write(str(metrics["MRR"]) + " " + str(n) + "\n")
            fout.write(str(metrics["NDCG@5"]) + " " + str(n) + "\n")
            fout.write(str(metrics["Hit@1"]) + " " + str(n) + "\n")
    dataloader.join()
    return metrics


def train(args):
    hvd = None
    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_json_file(args.config_file)

    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        _news_tsv_path(args),
        args,
        tokenizer
    )
    
    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory= get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
            news_abstract, news_abstract_type, news_abstract_attmask, \
            news_body, news_body_type, news_body_attmask, \
            news_category, news_domain, news_subcategory]
        if x is not None], axis=1)
    
    image_size = read_news_image_size(args.image_size_file)
    news_imageid_dict,news_roi_feature,news_roi_location,news_roi_mask,news_image_whole =read_news_image(args.roi_file,args.whole_file,image_size,args)

    news_id2image_id = _map_news_to_image(news_index, news_imageid_dict)

    model = mmrec(config,args)
    start_epoch=0
    if args.load_ckpt_name is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path,map_location='cpu')
        start_epoch = int(ckpt_path.split('-')[-1].split('.')[0])
        if hvd_rank == 0:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Model loaded from {ckpt_path}")
            logging.info(f"start from epoch [{start_epoch}]")
        del checkpoint
    
    model.train()

    finetuneset = build_finetuneset(config)
    for name,para in model.named_parameters():
        logging.info(name)
        para.requires_grad = _param_trainable(name, finetuneset)

    if args.enable_gpu:
        model = model.cuda()

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Average)


    dataloader = DataLoaderTrain(
        news_index=news_index,
        news_combined=news_combined,
        data_dir=os.path.join(args.root_data_dir,
                            f'{args.dataset}/{args.train_dir}'),
        filename_pat=args.filename_pat,
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=getattr(args, "enable_prefetch", False),
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
        news_imageid_dict=news_id2image_id,
        news_roi_feature=news_roi_feature,
        news_roi_location=news_roi_location,
        news_roi_mask=news_roi_mask,
        news_image_whole=news_image_whole
    )
    logging.info('Training...')

    world = max(1, args.hvd_size if args.enable_hvd else 1)
    args.max_steps_per_epoch = args.max_steps_per_epoch//(world*args.batch_size)

    val_dir_name = args.valid_dir or "dev"
    val_data_dir = os.path.join(args.root_data_dir, args.dataset, val_dir_name)
    sel_key = _selection_key(getattr(args, "selection_metric", "MRR"))
    best_score = -1.0
    best_epoch = -1
    best_metrics = None
    epoch_logs = []

    for ep in range(start_epoch,args.epochs):
        model.train()
        torch.set_grad_enabled(True)
        loss = 0.0
        if args.enable_hvd:
            hvd.join()
        for cnt, (news_feature, log_ids, log_mask, input_ids, targets,input_imgs,image_loc,image_mask) in enumerate(dataloader,start=1):
            if cnt > args.max_steps_per_epoch or (args.debug and cnt>10):
                break

            if args.enable_gpu:
                news_feature = news_feature.cuda(non_blocking=True)
                log_ids = log_ids.cuda(non_blocking=True)
                log_mask = log_mask.cuda(non_blocking=True)
                input_ids = input_ids.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                input_imgs=input_imgs.cuda(non_blocking=True)
                image_loc=image_loc.cuda(non_blocking=True)
                image_mask = image_mask.cuda(non_blocking=True)

            feature_dict = {
                'input_txt':torch.narrow(news_feature,1,0,args.num_words_title),
                'input_imgs':input_imgs,
                'image_loc':image_loc,
                'token_type_ids': torch.narrow(news_feature,1,args.num_words_title,args.num_words_title),
                'attention_mask':torch.narrow(news_feature,1,args.num_words_title*2,args.num_words_title),
                'image_attention_mask':image_mask
            }

            bz_loss, y_hat = model(feature_dict, input_ids, log_ids, log_mask, targets)
            loss += bz_loss.data.float()
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            batch_loss = bz_loss.clone().detach()
            avg = _reduce_mean(loss.clone().detach(), args.enable_hvd, hvd)
            avg_batch = _reduce_mean(batch_loss, args.enable_hvd, hvd)

            if ((cnt!=0 and cnt % args.log_steps == 0) or (cnt != 0 and args.debug)) and hvd_rank == 0:
                logging.info(
                    'epoch [{}] [{}] Ed: {}, train_avg_cumu_loss: {:.5f}, train_avg_batch_loss: {:.5f}'.format(
                        ep, hvd_rank, cnt * args.batch_size, avg.data/cnt, avg_batch.data ))

        ckpt_payload = {
            'model_state_dict': model.state_dict(),
            'category_dict': category_dict,
            'domain_dict': domain_dict,
            'subcategory_dict': subcategory_dict,
            'epoch': ep + 1,
        }
        if hvd_rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            torch.save(ckpt_payload, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

        val_metrics = None
        if os.path.isdir(val_data_dir):
            news_scoring_t, news_scoring_v = encode_news(
                args,
                model,
                news_combined,
                news_roi_feature,
                news_roi_location,
                news_roi_mask,
                news_image_whole,
                news_id2image_id,
            )
            val_metrics = score_impressions(
                args,
                model,
                news_index,
                news_scoring_t,
                news_scoring_v,
                news_id2image_id,
                val_data_dir,
                hvd_size,
                hvd_rank,
                hvd_local_rank,
            )
            logging.info(
                "val epoch {} MRR={:.6f} NDCG@5={:.6f} Hit@1={:.6f} (n={})".format(
                    ep + 1,
                    val_metrics["MRR"],
                    val_metrics["NDCG@5"],
                    val_metrics["Hit@1"],
                    val_metrics["n"],
                )
            )
            score = float(val_metrics[sel_key])
            if score > best_score:
                best_score = score
                best_epoch = ep + 1
                best_metrics = dict(val_metrics)
                if hvd_rank == 0:
                    best_path = os.path.join(args.model_dir, "best.pt")
                    ckpt_payload["val_metrics"] = val_metrics
                    ckpt_payload["selection_metric"] = sel_key
                    torch.save(ckpt_payload, best_path)
                    logging.info(
                        "best checkpoint updated: epoch {} {}={:.6f} → {}".format(
                            best_epoch, sel_key, best_score, best_path
                        )
                    )
        else:
            logging.warning("validation dir missing: %s (last epoch will be used as best)", val_data_dir)

        epoch_logs.append({"epoch": ep + 1, "val": val_metrics})

    if best_epoch < 0 and hvd_rank == 0:
        last_ep = args.epochs
        last_path = os.path.join(args.model_dir, f"epoch-{last_ep}.pt")
        best_path = os.path.join(args.model_dir, "best.pt")
        if os.path.isfile(last_path):
            import shutil
            shutil.copy2(last_path, best_path)
            best_epoch = last_ep
            logging.info("no val scores; copied %s → best.pt", last_path)

    summary = {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "selection_metric": sel_key,
        "epoch_logs": epoch_logs,
    }
    if hvd_rank == 0:
        log_json = os.path.join(args.log_dir, "val_epoch_log.json")
        with open(log_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info("val log → %s", log_json)
        if best_metrics:
            logging.info(
                "best val epoch={} {}={:.6f}  MRR={:.6f}  NDCG@5={:.6f}  Hit@1={:.6f}".format(
                    best_epoch,
                    sel_key,
                    best_score,
                    best_metrics["MRR"],
                    best_metrics["NDCG@5"],
                    best_metrics["Hit@1"],
                )
            )
    return summary


def test(args):

    hvd = None
    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        if ckpt_path is None and args.load_ckpt_name != "best.pt":
            ckpt_path = utils.get_checkpoint(args.model_dir, "best.pt")
    else:
        ckpt_path = utils.get_checkpoint(args.model_dir, "best.pt") or utils.latest_checkpoint(args.model_dir)

    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path,map_location='cpu')

    if 'subcategory_dict' in checkpoint:
        subcategory_dict = checkpoint['subcategory_dict']
    else:
        subcategory_dict = {}

    category_dict = checkpoint['category_dict']
    domain_dict = checkpoint['domain_dict']

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_json_file(args.config_file)
    model = mmrec(config,args)
    
    if args.enable_gpu:
        model.cuda()
    if hvd_rank == 0:
        model.load_state_dict(checkpoint['model_state_dict'])
    ckpt_epoch = checkpoint.get("epoch")
    ckpt_val = checkpoint.get("val_metrics")
    del checkpoint
    logging.info(f"Model loaded from {ckpt_path} epoch={ckpt_epoch} val={ckpt_val}")

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index = read_news_bert(
        _news_tsv_path(args),
        args,
        tokenizer,
        'test'
    )

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory= get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
            news_abstract, news_abstract_type, news_abstract_attmask, \
            news_body, news_body_type, news_body_attmask, \
            news_category, news_domain, news_subcategory]
        if x is not None], axis=1)

    image_size = read_news_image_size(args.image_size_file)
    news_imageid_dict,news_roi_feature,news_roi_location,news_roi_mask,news_image_whole =read_news_image(args.roi_file,args.whole_file,image_size,args)

    news_id2image_id = _map_news_to_image(news_index, news_imageid_dict)
    news_scoring_t, news_scoring_v = encode_news(
        args,
        model,
        news_combined,
        news_roi_feature,
        news_roi_location,
        news_roi_mask,
        news_image_whole,
        news_id2image_id,
    )
    test_data_dir = os.path.join(args.root_data_dir, f'{args.dataset}/{args.test_dir}')
    metrics = score_impressions(
        args,
        model,
        news_index,
        news_scoring_t,
        news_scoring_v,
        news_id2image_id,
        test_data_dir,
        hvd_size,
        hvd_rank,
        hvd_local_rank,
        result_name="test_result",
    )
    if hvd_size <= 1:
        with open(os.path.join(args.log_dir, "final_result.txt"), "w") as fout:
            fout.write("MRR\t{:.6f}\n".format(metrics["MRR"]))
            fout.write("NDCG@5\t{:.6f}\n".format(metrics["NDCG@5"]))
            fout.write("Hit@1\t{:.6f}\n".format(metrics["Hit@1"]))
        with open(os.path.join(args.log_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"ckpt": ckpt_path, "ckpt_epoch": ckpt_epoch, "val": ckpt_val, "test": metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )
        logging.info(
            "TEST MRR={:.6f}  NDCG@5={:.6f}  Hit@1={:.6f}  (n={})".format(
                metrics["MRR"],
                metrics["NDCG@5"],
                metrics["Hit@1"],
                metrics["n"],
            )
        )
    return metrics


if __name__ == "__main__":
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if 'cal' in args.mode:
        metric = [[0,0] for i in range(3)]
        for i in range(args.hvd_size):
            with open(os.path.join(args.log_dir,f'test_result_{i}.txt'),'r')as f:
                cnt = 0
                for line in f:
                    temp = line.split(' ')
                    avg = float(temp[0])
                    tot = float(temp[1])
                    sum_val = avg*tot
                    metric[cnt][0]+=sum_val
                    metric[cnt][1]+=tot
                    cnt+=1
        names = ["MRR", "NDCG@5", "Hit@1"]
        with open(os.path.join(args.log_dir,'final_result.txt'),'w') as fout :
            for i in range(3):
                fout.write(f"{names[i]}\t{metric[i][0]/metric[i][1]:.6f}\n")
        exit()
    if 'train' in args.mode:
        utils.setuplogger(os.path.join(args.log_dir,'log_train.txt'))
        logging.info(args)
        train(args)
    if 'test' in args.mode:
        utils.setuplogger(os.path.join(args.log_dir,'log_test.txt'))
        logging.info(args)
        test(args)


