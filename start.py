import os
from asyncio import Queue, create_task, run, gather, run_coroutine_threadsafe, get_event_loop
from multiprocessing import Process, Pipe, Queue as MpQueue
from threading import Thread
from argparse import ArgumentParser
# from io import BytesIO
# from datetime import datetime, timezone

# import cv2 as cv
# import numpy as np
from dotenv import load_dotenv
# for Test

# from paddleocr import OpenVinoPaddleOCR
from sync_v3 import ConnectProcess

# import time
# import PIL.Image as Image

from ultralytics.yolo.v8.detect.predict import predict

async def fetchWorker(queue):
    print("EXECUTED::fetchworker")
    # targetid = os.getenv("targetid")
    targetid = "6gZ8Xn5hl0110Eu8APM6UGgEBZ6HwZSbMpTg2ZYoqlo"
    domain = os.getenv("domain", default="NEXIVIL-dabeom.local")
    protocol = os.getenv("protocol", default="https")
    
    print("targetid", targetid)
    print("domain", domain)
    print("protocol", protocol)
    connect_process = ConnectProcess(targetid, domain, protocol)
    isMatch = connect_process.verify_pb_key()
    if isMatch is False:
        raise Exception("Pollution 'targetid' value.")
    connect_process.generate_ssl()

    print("SUCESS::sync")

    while True:
        data = await queue.get()
        print("GETDATA::from_fetch_queue", data is not None)
        if data is None:
            queue.task_done()
            continue
        try:
            if 'entry_date' in data:
                connect_process.fetch_with_target("cctv", "POST", data)
            else:
                connect_process.fetch_with_target("cctv", "PATCH", data)
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
    # cctv_rx, cctv_tx = Pipe(duplex=False)
    mp_queue = MpQueue()

    procs = []
    det_in_out_process = Process(target=predict, args=(mp_queue,))
    procs.append(det_in_out_process)
    # capture_lp_process = Process(target=process_capture, args=(cctv_rx, mp_queue,))
    # procs.append(capture_lp_process)

    det_in_out_process.start()
    # capture_lp_process.start()

    print(f'det_pid: {det_in_out_process.ident}')
    # print(f'cap_pid: {capture_lp_process.ident}')

    # tasks.append(create_task(process_video(rtsp, x1=x1, x2=x2, y1=y1, y2=y2, thickness=thickness)))

    # asyncio_queue = Queue()
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
    # tasks.append(create_task(tasksWorker(asyncio_queue, fetch_queue)))
    await gather(*tasks, return_exceptions=True)

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

    args = parser.parse_args()

    run(main())
