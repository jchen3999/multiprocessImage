import json
import os
from datetime import datetime
import time
from os.path import join
import re
from collections import defaultdict
import argparse
import colorama
from colorama import Fore, Style
import gpustat
import math


def get_args():
    parser = argparse.ArgumentParser(description='Parameters to pass to txpd_test.m')
    parser.add_argument('--kill', action='store_true', help='if specified, kill all matlab process')
    parser.add_argument('--check', action='store_true', help='if specified, check current progress')
    # parser.add_argument('--startnum', type=int, default=1)
    # parser.add_argument('--testnum', type=str, default='1:2')
    # parser.add_argument('--burstnum', type=str, default='1:30')
    # parser.add_argument('--RF_frames', type=int, default=200)
    # parser.add_argument('--angleCount', type=int, default=21)
    # parser.add_argument('--info_num', type=str, default='4-21-23/info21.mat')
    return parser.parse_args()


def parse_range_string(range_srt):
    """
    :param range_srt: a string specifying selected ranges; e.g., 1:20 23 25 30:40
    :return: list of each selected index
    """
    selected_index = []
    dummy = range_srt.split()
    for d in dummy:
        if ':' in d:
            start, end = d.split(':')
            selected_index += list(range(int(start), int(end) + 1))
        else:
            selected_index.append(int(d))
    return selected_index


def get_cache_dir(args):
    """
    :param args: args to store run configs
    :return:  None. get cache dir to store temp scripts
    """
    os.makedirs('./cache', exist_ok=True)
    base = str(datetime.now()).split(' ')[0].replace('-', '_') + '_run'
    current_runs = os.listdir('./cache')
    current_runs = len([run for run in current_runs if run.startswith(base)]) + 1
    args.cache_dir = join('./cache', base + f'_{current_runs}')
    return


def concise_index(index_list):
    """
    :param index_list: list of index; e.g., [1,2,3,5,6,7]
    :return: a string of readable ranges; e.g., '1 --> 3, 5-->7'
    """
    index_list = sorted(index_list)
    index_start_end = []
    cur_start = None
    prev = None
    for idx in index_list:
        if not cur_start:
            cur_start = idx
            prev = idx
        elif idx == prev + 1:
            prev = idx
            continue
        else:
            cur_end = prev
            index_start_end.append(f'{cur_start} --> {cur_end}')
            cur_start = idx
            prev = idx
    cur_end = index_list[-1]
    index_start_end.append(f'{cur_start} --> {cur_end}')
    return ', '.join(index_start_end)


def get_tests(date_path):
    """
    :param date_path: path to the test folders (usually exp date)
    :return: indexes of all test folders there
    """
    tests = os.listdir(date_path)
    tests = [d for d in tests if d.startswith('test')]
    not_empty_tests = []
    for test in tests:
        bursts = os.listdir(join(date_path, test))
        bursts = [b for b in bursts if b.startswith('burst') and b.endswith('.mat')]
        if len(bursts) == 0:
            print(Fore.RED + f"detected empty test folder with no burst: {test}." + Style.RESET_ALL)
        else:
            not_empty_tests.append(test)
    tests = [int(d.split('test')[1]) for d in not_empty_tests]
    return sorted(tests)


def get_bursts(test_path):
    """
    :param test_path: path to the burst folder
    :return: indexes of all bursts there
    """
    bursts = os.listdir(test_path)
    bursts = [file for file in bursts if file.startswith('burst') and file.endswith('.mat')]
    bursts = [int(file.split('burst')[1].split('.')[0]) for file in bursts]
    return sorted(bursts)


def get_test_burst_num(date_path):
    """
    :param date_path: path to the test folders (usually exp date)
    :return: indexes of all test folders and number of bursts there
    """
    test_indexes = get_tests(date_path)
    test_folders = [f'test{idx}' for idx in test_indexes]
    idx_ranges = concise_index(test_indexes)
    print(f"{Fore.GREEN}")
    print(f"detected tests: {idx_ranges}")
    num_bursts = sum([len(get_bursts(os.path.join(date_path, t))) for t in test_folders])
    print(f"detected bursts: {num_bursts} bursts"
          f"{Style.RESET_ALL}")

    return test_indexes, test_folders


def get_date_path(args):
    """
    :param args: args to store run configs
    :return: None. collect exp date and update args
    """
    while True:
        args.date = input(
            f"{Fore.BLUE}which date? \n  {Fore.GREEN} "
            f"e.g., 4-21-23 {Style.RESET_ALL}\n"
        )
        date_path = os.path.join(args.dataset_root, args.date)
        if os.path.exists(date_path):
            break
        else:
            print(Fore.RED + f'path {date_path} does not exist!' + Style.RESET_ALL)
    args.date_path = date_path
    return


def get_test(test_indexes, args):
    """
    :param test_indexes: all detected test indexes under an exp folder
    :param args: args to store run configs
    :return: None. collect selected test indexes and update args
    """
    while True:
        selected_test_indexes = input(
            f"{Fore.BLUE}which tests? {Fore.GREEN} \n"
            f"use space ' ' to separate multiple test indexes/ranges; e.g., 1 2 3 4 5 \n"
            f"use ':' to specify a range (nonexistent indexes will be ignored); e.g., 1:10 13:20 23 \n"
            f"use 'all' to run all tests; e.g., all"
            f"{Style.RESET_ALL}\n"
        )
        if selected_test_indexes == 'all':
            args.tests = test_indexes
            break
        else:
            try:
                selected_idxs = []
                dummy = selected_test_indexes.split()
                for d in dummy:
                    if ':' in d:
                        start, end = d.split(':')
                        selected_idxs += list(range(int(start), int(end) + 1))
                    else:
                        selected_idxs.append(int(d))
                args.tests = selected_idxs
                break
            except:
                print(
                    Fore.RED + f"an error occurred while parsing test indexes '{selected_test_indexes}'" + Style.RESET_ALL)
    for idx in args.tests:
        if idx not in test_indexes:
            print(
                Fore.RED + f"ignoring test{idx}, which does not exist in {os.path.join(args.dataset_root, args.date)} or has no burst data within" + Style.RESET_ALL
            )
            args.tests.remove(idx)
    print(f'the following tests will be scheduled to run: {concise_index(args.tests)}')
    return


def get_burst(args):
    """
    :param args: args to store run configs
    :return:  None. collect selected burst indexes and update args
    """
    args.bursts = defaultdict(list)
    run_all_burst = input(
        f"{Fore.BLUE}do you want to run all bursts from all tests? \n  {Fore.GREEN} "
        f"enter one of the following y / yes / n / no \n"
        f"{Style.RESET_ALL}\n"
    )
    if run_all_burst.startswith('y'):
        for t_idx in args.tests:
            args.bursts[t_idx] = get_bursts(os.path.join(args.date_path, f'test{t_idx}'))
    else:
        all_same_range = input(
            f"{Fore.BLUE}do you want to use the same burst range for all test (nonexistent indexes will be ignored)? \n  "
            f"{Fore.GREEN}"
            f"enter one of the following y / yes / n / no"
            f"{Style.RESET_ALL}\n"
        )
        if all_same_range.startswith('y'):
            burst_range = input(
                f"{Fore.BLUE}which burst range? \n  {Fore.GREEN} "
                f"e.g., 1:10"
                f"{Style.RESET_ALL}\n"
            )
            for t_idx in args.tests:
                args.bursts[t_idx] = parse_range_string(burst_range)
        else:
            print(f'{Fore.BLUE}please manually enter the burst range for each selected burst: \n')
            f"use space ' ' to separate multiple burst indexes/ranges; e.g., 1 2 3 4 5 \n"
            f"use ':' to specify a range (nonexistent indexes will be ignored); e.g., 1:10 13:20 23 \n"

            for t_idx in args.tests:
                cur_index = input(
                    f"test{t_idx}: {Fore.BLUE} which burst range? \n  {Fore.GREEN} "
                    f"existing bursts are: {concise_index(get_bursts(os.path.join(args.date_path, f'test{t_idx}')))}\n"

                )
                args.bursts[t_idx] = parse_range_string(cur_index)

    for t_idx in sorted(args.bursts.keys()):
        args.bursts[t_idx] = sorted(list(set(args.bursts[t_idx])))
        non_exist_bursts = []
        for b_idx in args.bursts[t_idx]:
            if not os.path.exists(os.path.join(args.date_path, f'test{t_idx}', f'burst{b_idx}.mat')):
                print(
                    Fore.RED + f"ignoring test{t_idx}/burst{b_idx}.mat, which does not exist in {os.path.join(args.dataset_root, args.date)}" + Style.RESET_ALL
                )
                non_exist_bursts.append(b_idx)
        for b_idx in non_exist_bursts:
            args.bursts[t_idx].remove(b_idx)
        if not args.bursts[t_idx]:
            del args.bursts[t_idx]

    print(f'done. the following tests/bursts are scheduled to process:\n')
    for t_idx in sorted(args.bursts.keys()):
        print(f"test{t_idx}: {concise_index(args.bursts[t_idx])}")
    return


def get_devices(args):
    print(f'{Fore.BLUE}current gpu status:')
    gpustat.print_gpustat()
    gpu_indexes = input(
        f"{Fore.BLUE}which gpus to use? \n  {Fore.GREEN} "
        f"please check the gpu status above and select available one(s); e.g., 0\n"
        f"use space ' ' to separate multiple gpu indexes; e.g., 0 1 2 3 "
        f"{Style.RESET_ALL}\n"
    )
    gpu_indexes = [int(idx) for idx in gpu_indexes.split()]
    args.gpu_indexes = gpu_indexes
    return


def get_num_jobs(args):
    num_jobs = input(
        f"{Fore.BLUE}how many bursts to process simultaneously \n  {Fore.GREEN}"
        f"note that each job will take up ~7% of the total memory\n"
        f"maximum capacity: 12; recommend value: <= 8 "
        f"{Style.RESET_ALL}\n"
    )
    args.num_jobs = int(num_jobs)
    return


def get_RF_frames(args):
    RF_frames = input(
        f"{Fore.BLUE}RF_frames? \n  {Fore.GREEN}"
        f"e.g.; 200"
        f"{Style.RESET_ALL}\n"
    )
    args.RF_frames = int(RF_frames)
    return


def get_angleCount(args):
    angleCount = input(
        f"{Fore.BLUE}angleCount? \n  {Fore.GREEN}"
        f"e.g.; 21"
        f"{Style.RESET_ALL}\n"
    )
    args.angleCount = int(angleCount)
    return


def get_infoMat(args):
    while True:
        infoMat = input(
            f"{Fore.BLUE}which infoMat? \n  {Fore.GREEN}"
            f"e.g.; 4-21-23/info21.mat"
            f"{Style.RESET_ALL}\n"
        )
        if not os.path.exists(f'/gpfs/fs2/scratch/mdoyley_lab/fUS/channel_data/{infoMat}'):
            print(
                Fore.RED + f'infoMat not found at /gpfs/fs2/scratch/mdoyley_lab/fUS/channel_data/{infoMat}!' + Style.RESET_ALL)
        else:
            break
    args.infoMat = infoMat
    return


def get_th(args):
    th = input(
        f"{Fore.BLUE}th? \n  {Fore.GREEN}"
        f"e.g.; 50"
        f"{Style.RESET_ALL}\n"
    )
    args.th = int(th)
    return


def get_schedule(args):
    print(f"scheduling runs...")
    job_per_gpu = []
    for _ in range(len(args.gpu_indexes) - 1):
        job_per_gpu.append(math.ceil(args.num_jobs / len(args.gpu_indexes)))
    job_per_gpu.append(args.num_jobs - sum(job_per_gpu))

    num_bursts = 0
    flattened_burst_list = []
    for test in args.tests:
        num_bursts += len(args.bursts[test])
        flattened_burst_list += [f"{test}_{b_idx}" for b_idx in args.bursts[test]]
    args.flattened_burst_list = flattened_burst_list
    bursts_per_job = math.ceil(num_bursts / args.num_jobs)

    job_lists = []
    for i in range(args.num_jobs - 1):
        cur_job_list = flattened_burst_list[i * bursts_per_job:(i + 1) * bursts_per_job]
        job_lists.append(cur_job_list)
    cur_job_list = flattened_burst_list[(args.num_jobs - 1) * bursts_per_job:]
    job_lists.append(cur_job_list)

    # parse job_lists to matlab nested arrays:
    def parse_job_list(cur_job_list):
        # [test_burst] -> {{test,burst}}
        mat_list = []
        for test_burst in cur_job_list:
            cur_test, cur_burst = test_burst.split('_')
            mat_list.append(f"{'{'}{cur_test},{cur_burst}{'}'}")

        matlab_job_list = '{' + ','.join(mat_list) + '}'
        return matlab_job_list

    matlab_job_lists = [parse_job_list(cur_job_list) for cur_job_list in job_lists]
    args.matlab_job_lists = matlab_job_lists
    return


def generate_m_files(args):
    """
    :param args: args to store run configs
    :return: None. Save m files to cache
    """
    print(f"generating scripts...")
    os.makedirs(join(args.cache_dir, 'scripts'), exist_ok=True)
    os.makedirs(join(args.cache_dir, 'logs'), exist_ok=True)
    for idx, job_list in enumerate(args.matlab_job_lists):
        cur_m_code = f"""startnum=1;
            date='{args.date}';
            date_tmp=date;
            test_burst_pairs = {job_list};
            for pair_idx=1:length(test_burst_pairs) % todo:
                try
                    testnum = test_burst_pairs{'{pair_idx}{1}'};
                    burstnum = test_burst_pairs{'{pair_idx}{2}'};
                    RF_frames={args.RF_frames}; % todo
                    angleCount = {args.angleCount}; % todo
                    addpath('/gpfs/fs2/scratch/mdoyley_lab/fUS/codes/SVD');
                    addpath(genpath('/gpfs/fs2/scratch/mdoyley_lab/fUS/codes/oldCodes'))
                    addpath('/gpfs/fs2/scratch/mdoyley_lab/fUS/codes/beamformedPD');
                    load('/gpfs/fs2/scratch/mdoyley_lab/fUS/channel_data/{args.infoMat}'); % todo: info_num
                    date=date_tmp;
                    test=['test',num2str(testnum+startnum-1)];

                    burst=burstnum;
                    FullBoard128 = 1;
                    fname = ['/scratch/mdoyley_lab/fUS/channel_data/' date '/' test '/burst',num2str(burst) '.mat'];
                    a=load(fname);
                    b=zeros(size(a.RcvData));
                    a=cat(3,a.RcvData,b);
                    RcvData{'{2}'}=a;
                    ASAP_size = 4;
                    ASAP_id = 2;
                    receiveSubBF;
                    RFa = RF;
                    clear RF channel_data RcvData
                    load('/gpfs/fs2/scratch/mdoyley_lab/fUS/channel_data/{args.infoMat}'); % todo:
                    %
                    date=date_tmp;
                    fname = ['/scratch/mdoyley_lab/fUS/channel_data/' date '/' test '/burst',num2str(burst) '.mat'];
                    a=load(fname);
                    b=zeros(size(a.RcvData));
                    a=cat(3,a.RcvData,b);
                    RcvData{'{2}'}=a;
                    ASAP_size = 4;
                    ASAP_id = 1;
                    receiveSubBF;
                    RFb = RF;
                    %disp(ASAP_id);
                    % compounding
                    if angleCount == 0
                        angleCount = numAngles;
                    end
                    ha = angleCount/2;
                    for i = 1:numRFframes
                        setstart = (i-1)*numAngles;
                        i1 = setstart + (numAngles/2) - ha  + (1:angleCount);
                        RF1(:,:,i) = sum(RFa(:,:,i1),3);
                        RF2(:,:,i) = sum(RFb(:,:,i1),3);
                    end

            %         % IQ formation
                    fs = 1540/(2*(z(2)-z(1)));
                    IQ1 = RF2IQ(RF1,Trans.frequency*1e6,fs);
                    IQ2 = RF2IQ(RF2,Trans.frequency*1e6,fs);


                    th={args.th}; % todo:
                    [~,BloodIQ1] = computeBloodSignal(IQ1,th);
                    [~,BloodIQ2] = computeBloodSignal(IQ2,th);

                    R = sum(BloodIQ1.*conj(BloodIQ2),3);
                    ang = angle(R);
                    ko = pi/3;
                    PD_frameSplit_corr = abs(R).*exp(-(ang.^2)./(ko.^2));
                    PD_frameSplit = abs(R);

            %         PD_frameSplit_corr = 10*log10(PD_frameSplit_corr./max(PD_frameSplit_corr(:)));
            %         PD_frameSplit = 10*log10(PD_frameSplit./max(PD_frameSplit(:)));

                    [zz, xx] = ndgrid(z,x);
                    x_new = linspace(x(1),x(end),length(z));
                    [zz_new,xx_new] = ndgrid(z,x_new);

                    F1 = griddedInterpolant(zz,xx,PD_frameSplit_corr,'cubic');
                    F2 = griddedInterpolant(zz,xx,PD_frameSplit,'cubic');

                    PD_frameSplit_corr = F1(zz_new,xx_new);
                    PD_frameSplit = F2(zz_new,xx_new);
            %
                    loc=['/gpfs/fs2/scratch/mdoyley_lab/fUS/TSAPdata/']; % todo: save_dir
                    loc = [loc date '/'];
                    if exist(loc,'dir') ~= 7
                        mkdir(loc)
                    end
                    loc = [loc num2str(test) '/'];
                    if exist(loc,'dir') ~= 7
                        mkdir(loc)
                    end
                    save([loc 'burst' num2str(burstnum),'.mat'],'PD_frameSplit','PD_frameSplit_corr');
                    disp([num2str(burstnum), '/',num2str(startnum+testnum-1),'/' date]);
                catch
                    warning('fail to process this one:');
                    warning([num2str(burstnum), '/',num2str(startnum+testnum-1),'/' date]);
                    warning('ignoring errors, processing the next one');
                end
            end
            quit();
            """
        with open(join(args.cache_dir, 'scripts', f'part_{idx}.m'), 'w') as f:
            f.write(cur_m_code)
        cur_bash_code = f"""module load matlab/r2022a;\nCUDA_VISIBLE_DEVICES={args.gpu_indexes[idx % len(args.gpu_indexes)]} matlab -nodesktop -r part_{idx} | tee {join('../logs', f'part_{idx}_log.txt')} &>/dev/null &"""
        with open(join(args.cache_dir, 'scripts', f'part_{idx}.sh'), 'w') as f:
            f.write(cur_bash_code)
    return


def run_jobs(args):
    args_dict = vars(args)
    with open(join(args.cache_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f)
    print(f"running jobs...")
    os.chdir(join(args.cache_dir, 'scripts'))
    for idx in range(args.num_jobs):
        cur_script_pth = f'part_{idx}.sh'
        os.system(f"bash {cur_script_pth}")
        time.sleep(0.5)
    print(f"jobs are running in background.")
    print(f"config files are stored in {args.cache_dir}; including temporary matlab/bash files and logs")
    print(f"- to kill these jobs, run {Fore.BLUE}'python processImages.py --kill'{Style.RESET_ALL}")
    print(
        f"- to check current progress, run {Fore.BLUE}'python processImages.py --check'{Style.RESET_ALL}; then enter this run_name '{os.path.basename(args.cache_dir)}' as prompted")
    print(
        f"- to check current system info, run {Fore.BLUE}'top'{Style.RESET_ALL}; use {Fore.BLUE}'Ctrl-c'{Style.RESET_ALL} to exit")
    print(
        f"- to check current gpu info, run {Fore.BLUE}'watch gpustat'{Style.RESET_ALL}; use {Fore.BLUE}'Ctrl-c'{Style.RESET_ALL} to exit")
    return


if __name__ == '__main__':
    colorama.init()
    args = get_args()
    args.dataset_root = '/scratch/mdoyley_lab/fUS/channel_data'
    if args.kill:
        print(Fore.RED + f"note that this will kill all matlab related jobs running on current node." + Style.RESET_ALL)
        confirmed = input("to confirm, enter y / yes\n")
        if confirmed.startswith('y'):
            os.system("ps -A | grep MATLAB | awk '{print $1}' | xargs kill -9 $1")
            os.system("ps -A | grep tee | awk '{print $1}' | xargs kill -9 $1")
    elif args.check:
        current_runs = os.listdir('./cache')
        current_runs = sorted([run for run in current_runs if '_run_' in run],
                              key=lambda x: int(''.join(x.replace('_run_', '_').split('_'))))
        latest_run = current_runs[-1]
        print(f"The latest run is {latest_run}.")
        confirmed = input("use this one? enter y / yes / n /no\n")
        if confirmed.startswith('y'):
            use_run = latest_run
        else:
            print(f"All runs are: {current_runs}.")
            use_run = input(f"which one to check?\n")
        with open(join('./cache', use_run, 'args.json'), 'r') as f:
            configs = json.load(f)
        num_all_burst = len(configs['flattened_burst_list'])
        all_logs = os.listdir(join(configs['cache_dir'], 'logs'))
        all_logs_text = []
        for log in all_logs:
            with open(join(configs['cache_dir'], 'logs', log), 'r') as f:
                dummy = f.readlines()
            all_logs_text += dummy
        pattern = r"\d+/[-+]?\d+/\d{1,2}-\d{1,2}-\d{2,4}$"
        matches = [line.strip() for line in all_logs_text if re.search(pattern, line)]
        print(f"{len(matches)}/{num_all_burst} bursts have been processed.")

    else:
        get_cache_dir(args)
        print(f"""
        if you enter something wrong, use 'ctrl-c' to stop this script
        """)
        get_date_path(args)
        test_indexes, test_folders = get_test_burst_num(args.date_path)
        get_test(test_indexes, args)
        get_burst(args)
        get_devices(args)
        get_num_jobs(args)
        get_RF_frames(args)
        get_angleCount(args)
        get_infoMat(args)
        get_th(args)
        get_schedule(args)
        generate_m_files(args)
        run_jobs(args)
