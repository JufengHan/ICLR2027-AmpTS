from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4,
    PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader,
    SWATSegLoader, UEAloader, Dataset_DiffusionTS, Dataset_nodate,
    Dataset_PEMS, Dataset_npy, Dataset_Solar, Dataset_ILI, Dataset_wind,
)
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'ILI': Dataset_ILI,
    'ili': Dataset_ILI,
    'illness': Dataset_ILI,
    'stock': Dataset_DiffusionTS,
    'energy': Dataset_DiffusionTS,
    'wind': Dataset_wind,
    'npy': Dataset_npy,
    'Nodate': Dataset_nodate,
    'PEMS': Dataset_PEMS,
    'Solar': Dataset_Solar,
    'solar': Dataset_Solar,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
}



def data_provider(args, flag):
    if args.data not in data_dict:
        raise KeyError(
            f"Unknown dataset type '{args.data}'. "
            f"Available types: {sorted(data_dict.keys())}"
        )

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if args.data == 'wind' or str(args.data_path).lower() == 'wind.csv':
        shuffle_flag = str(flag).lower() == 'train'
    else:
        shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False,
        )
        return data_set, data_loader

    if args.task_name == 'classification':
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
        )
        return data_set, data_loader

    if args.data == 'm4':
        drop_last = False

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader