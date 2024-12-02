from asyncio import Queue, create_task, run, gather, run_coroutine_threadsafe, get_event_loop
from multiprocessing import Process, Pipe, Queue as MpQueue
from threading import Thread
from argparse import ArgumentParser

from dotenv import load_dotenv

from sync_v3 import ConnectProcess

from ultralytics.yolo.v8.detect.predict import predict

async def fetchWorker(queue):
    print("EXECUTED::fetchworker")
    # targetid = os.getenv("targetid", default="6gZ8Xn5hl0110Eu8APM6UGgEBZ6HwZSbMpTg2ZYoqlo")
    # domain = os.getenv("domain", default="NEXIVIL-dabeom.local")
    # protocol = os.getenv("protocol", default="https")

    # targetid = args.targetid
    # protocol = args.protocol
    # domain = args.domain
    
    # print("targetid", targetid)
    # print("domain", domain)
    # print("protocol", protocol)
    # connect_process = ConnectProcess(targetid, domain, protocol)
    # isMatch = connect_process.verify_pb_key()
    # if isMatch is False:
    #     raise Exception("Pollution 'targetid' value.")
    # connect_process.generate_ssl()

    # print("SUCESS::sync")
    with open("./test.txt", "a",encoding='utf-8') as myfile:
        while True:
            data = await queue.get()
            print("GETDATA::from_fetch_queue", data is not None)
            if data is None:
                queue.task_done()
                continue
            try:
                _machine_number=data.get('machine_number',"")
                machine_number="인식불가" if _machine_number.strip()=="" else _machine_number
                print(_machine_number,machine_number)
                entry_date=data.get('entry_date',"")
                exit_date=data.get('exit_date',"")
                myfile.write(f'{machine_number} {entry_date or "-"} {exit_date or "-"}\n')

                if 'entry_date' in data:
                    with open(f'./testimg/full_in_{entry_date}_{machine_number}.jpg', "wb") as file:
                        file.write(data.get('full_image'))
                    with open(f'./testimg/lp_in_{entry_date}_{machine_number}.jpg', "wb") as file:
                        file.write(data.get('cropped_image'))
                else:
                    with open(f'./testimg/full_out_{exit_date}_{machine_number}.jpg', "wb") as file:
                        file.write(data.get('full_image'))
                    with open(f'./testimg/lp_out_{exit_date}_{machine_number}.jpg', "wb") as file:
                        file.write(data.get('cropped_image'))
              
                    
            except Exception as e:
                print(e)
            queue.task_done()

def queue_bridge(mp_queue, fetch_queue, loop, stop_loop):
    while True:
        item = mp_queue.get()
        if stop_loop:  # Check for the sentinel to end the loop
            break
        if item is not None:
            print("SEND::mp_queue->fetch_queue")
            run_coroutine_threadsafe(fetch_queue.put(item), loop=loop)


'''
WIP: Mustbe Initialized for each worker(CPU-bound)
'''


async def main():
    print("EXECUTED::main")

    '''
    WIP: Migrate redis(dragonfly) or any Task-Queue system and then Apply multiprocessing
    Piping bwtween workers and Main Process
    '''
    mp_queue = MpQueue()

    procs = []
    det_in_out_process = Process(target=predict, args=(mp_queue, args.rtsp, args.exit_direct))
    procs.append(det_in_out_process)

    det_in_out_process.start()

    print(f'det_pid: {det_in_out_process.ident}')

    fetch_queue = Queue()

    # Get the current event loop
    loop = get_event_loop()
    stop_loop = False
    # Start the queue bridge thread
    bridge_thread = Thread(target=queue_bridge, args=(
        mp_queue, fetch_queue, loop, stop_loop), daemon=True)
    bridge_thread.start()
    print(f'pid: {bridge_thread.ident}')

    tasks = []
    tasks.append(create_task(fetchWorker(fetch_queue)))
    await gather(*tasks, return_exceptions=True)
    print("AA")
    stop_loop = True
    mp_queue.put_nowait(None)
    for t in tasks:
        if t.done() and t.exception():
            continue
        t.cancel()

    for proc in procs:
        proc.shutdown()
    bridge_thread.join()

    return


if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser(
        prog='det_ocr',
        description='',
        epilog='')
    parser.add_argument('-w', '--worker', default=1, type=int)
    parser.add_argument("--rtsp", type=str, required=False, default="rtsp://admin1:adt@6400@223.48.4.67:554/h264-2")
    parser.add_argument("--targetid", type=str, required=False, default="sXfH4mrlfE5ltACJ6gzv0RqvycXvj0l6swPZB6yHQL4")
    parser.add_argument("--protocol", type=str, required=False, default="https")
    parser.add_argument("--domain", type=str, required=False, default="desktop-dc06siv.local")
    parser.add_argument("--exit_direct", type=str, required=False, default="right")
    
    args = parser.parse_args()

    run(main())
