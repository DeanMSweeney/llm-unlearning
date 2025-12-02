



def main(model_path_or_name: str, num_epochs: int = 5, is_mlm: bool = True, k: int = 10000, 
        sim_batch_size: int = -1, use_advantaged_for_grad: bool=True, agg_input: bool=True, 
        proportion_dev=0.75, do_dynamic_gradient_selection: bool=False, 
        lr: float = 1e-5, momentum: float = 0.9, batch_size: int = 16, seed: int = 89793, 
        num_workers: int = 4, start_at_epoch: int = 0, dedupe=''): 
    logger.info(f'Seed is {seed}')
    set_random_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_mlm: 
        if 'roberta' in model_path_or_name: 
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, add_prefix_space=True)
        else: 
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN)
        raise ValueError('No non-mlms currently')
    model = AutoModelForPreTraining.from_pretrained(model_path_or_name)
    model.resize_token_embeddings(len(tokenizer))

    model.train()
    model.to(device)

    dataset = WGDataset('../data/wg.tsv', '../data/wg_stats.tsv', tokenizer)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=WGDataset.collate_batch_creator(tokenizer), 
                                num_workers=num_workers, 
                                )
    optimizer = optim.SGD(model.parameters(), lr=lr)

    agg_dim = -1 if agg_input else -2
    new_grad_calc = _minimize_grads_2 if use_advantaged_for_grad else _maximize_grads_1

    logger.info('Retraining now')

    retrain(model, tokenizer, optimizer, device, dataloader, batch_size, is_mlm=is_mlm, k=k, 
            num_epochs=num_epochs, sim_batch_size=sim_batch_size, new_grad_calc=new_grad_calc, 
            proportion_dev=proportion_dev, do_dynamic_gradient_selection=do_dynamic_gradient_selection, 
            agg_dim=agg_dim, start_at_epoch=start_at_epoch, dedupe=dedupe, model_name=model_path_or_name)

if __name__=='__main__': 
    parser = argparse.ArgumentParser(description = 'This script uses prompts to sample sentences from an LM')
    parser.add_argument('-m', type=str, required=True, dest='model_path_or_name', help='path to the model or name of the model')
    parser.add_argument('-l', type=float, default=1e-5, dest='lr', help='learning rate')
    parser.add_argument('-k', type=int, default=10000, dest='k', help='the k in top k')
    parser.add_argument('--use-full-grad', dest='k', action='store_const', const=None, help='to use the full gradient (rather than k parts)') # if this and also -k are used then whichever is rightmost argument wins
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-n', type=int, default=5, dest='num_epochs', help='number of epochs to train for (total)')
    parser.add_argument('-b', type=int, default=16, dest='batch_size', help='batch size')
    parser.add_argument('--start-at', type=int, default=0, help='start at checkpoint epoch number (e.g., 1, and if training 5 epochs then 4 more epochs will be done)')
    parser.add_argument('--dedupe', type=str, default='', help='dedupe string (basically just the name of the experiment), models will be saved to `sim_checkpoints/{dedupe}/model_{epoch}`')
    parser.add_argument('--output-agg', dest='aggregation', action='store_const', const='output', default='input', help='to use output aggregation (default: input aggregation)')
    parser.add_argument('--dynamic_gradient_selection', dest='dynamic_gradient_selection', action='store_true', default=False, help='to choose disadvantaged and advantaged dynamically (default: static based on WG)')
    parser.add_argument('--use-disadvantaged', dest='use_advantaged_for_grad', action='store_false', default=True, help='to take gradient step to maximize disadvantaged (default: minimize advantaged)')
    parser.add_argument('--use-same-params', dest='sim_batch_size', action='store_const', const=None, default=-1, help='to use the same params each epoch (default: picks params each batch)')

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    is_mlm = 'bert' in args.model_path_or_name
    sim_batch_size = args.sim_batch_size
    use_advantaged_for_grad = args.use_advantaged_for_grad
    agg_input = args.aggregation=='input'
    if args.dynamic_gradient_selection: 
        direction_selection = 'dynamic'
    elif args.use_advantaged_for_grad: 
        direction_selection = 'adv'
    else: 
        direction_selection = 'disadv'

    if args.k is None: 
        partition_usage = 'full_grad'
    elif args.sim_batch_size is None: 
        partition_usage = 'all'
    else: 
        partition_usage = 'notall'

    dedupe_model_name = args.model_path_or_name.split('/')[-1]
    dedupe = f"{dedupe_model_name}/{partition_usage}/{'inp' if agg_input else 'outp'}/{direction_selection}/{args.lr}/64/{args.k}"
    main(args.model_path_or_name, num_epochs=args.num_epochs, is_mlm=is_mlm, k=args.k, 
        proportion_dev=0.5, do_dynamic_gradient_selection=args.dynamic_gradient_selection, 
        sim_batch_size=sim_batch_size, use_advantaged_for_grad=use_advantaged_for_grad, agg_input=agg_input, 
        lr=args.lr, momentum=args.momentum, batch_size=args.batch_size, start_at_epoch=args.start_at, dedupe=dedupe)

