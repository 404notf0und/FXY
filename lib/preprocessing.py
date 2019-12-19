# coding=utf-8
import urllib
import os
import json
import time
import logging
import logging.handlers
from threading import Thread
from multiprocessing import Process
from collections import defaultdict

def clean_rn(rn):
    """ clean \r\n of every string
    Parameters
    ----------
    rn:str
        string of data

    Returns
    ----------
    result:str
        no \r\n in string

    """
    return rn.strip().replace('\n', '').replace('\r', '').strip()

def clean_repeat(rp):
    """ clean repeat of list
    Parameters
    ----------
    rp:list
        list of data

    Returns
    ----------
    result:list
        no repeat in list
    """
    return list(set(rp))

def urlparse(url):
    """converting url encoded data to simple string
    Parameters
    ----------
    url:str
        string of url

    Returns
    ----------
    unquote_url:str
        unquote of url

    """
    unquote_url=urllib.parse.unquote(url)
    return unquote_url



def set_logger(filename, logmod):
    try:
        log_size = 100000000
        log_backupcount = 1

        handler = logging.handlers.RotatingFileHandler(filename, maxBytes=log_size, backupCount=log_backupcount)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt='[%b %d %H:%M:%S]')
        handler.setFormatter(formatter)
        logger = logging.getLogger(logmod)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        return logger
    except IOError:
        return logging


root_path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(root_path, "log", "cuckoo2txt.log")
alogger = set_logger(log_path, "CUCKOO2TXT")
"""
Security Scene: Malware Behaviour Detection
Class Cuckoo2Txt Function:  Extract malware api and other params from report.json of Cuckoo Sandbox,convert to .txt
"""
class Cuckoo2Txt(object):
    def __init__(self, report_path, dst_path):
        self.report_path = report_path
        self.dst_path = dst_path

    def _convert_arguments_full(self, arguments):
        arg_parts = list()
        for val in arguments.values():
            if not isinstance(val, str):
                arg_parts.append(str(val))
            else:
                arg_parts.append(json.dumps(val))
        return arg_parts

    def _convert_arguments_simplify(self, arguments):
        arg_parts = list()
        for val in arguments.values():
            # remove address such as 0xffffffff
            try:
                int(val, 16)
                continue
            except:
                pass
            if not isinstance(val, str):
                arg_parts.append(str(val))
            else:
                arg_parts.append(json.dumps(val))
        return arg_parts

    def convert_thread(self, pid, tid, api_calls):
        first_line = "# pid %s tid %s" % (pid, tid)
        lines = [first_line]
        for api_call in api_calls:
            parts = []
            arguments = api_call['arguments']
            category = api_call['category']
            api = api_call['api']
            parts.append(category)
            parts.append(api)
            parts.extend(self._convert_arguments_simplify(arguments))
            try:
                line = " ".join(parts)
            except:
                for part in parts:
                    print(part)
            lines.append(line)
        return "\n".join(lines)+"\n"

    def convert(self, filepath):
        try:
            with open(filepath, "r") as f:
                convert_txt = ""
                raw = f.read()
                report = json.loads(raw)

                pid_sequences = []
                tid_sequences = defaultdict(list)
                processes = {}
                procs = report["behavior"]["processes"]
                for proc in procs:
                    process_id = proc['pid']
                    parent_id = proc['ppid']
                    process_name = proc['process_name']
                    calls = proc['calls']
                    pid_sequences.append(process_id)
                    threads = {}
                    for call in calls:
                        thread_id = call['tid']
                        if thread_id not in tid_sequences[process_id]:
                            tid_sequences[process_id].append(thread_id)
                        try:
                            threads[thread_id].append(call)
                        except:
                            threads[thread_id] = []
                            threads[thread_id].append(call)
                    processes[process_id] = {}
                    processes[process_id]["parent_id"] = parent_id
                    processes[process_id]["process_name"] = process_name
                    processes[process_id]["threads"] = threads

                for pid in pid_sequences:
                    for tid in tid_sequences[pid]:
                        convert_txt += self.convert_thread(pid, tid, processes[pid]["threads"][tid])

                ori_name = os.path.basename(filepath)
                filename = ori_name.replace("json", "txt")
                dst_path = os.path.join(self.dst_path, filename)
                with open(dst_path, "w") as dst_f:
                    dst_f.write(convert_txt)
        except:
            print(filepath)
            print("convert error")

    def deal_thread(self, func):
        alogger.info("=======start convert cuckoo to txt=======")
        thlist = []
        max_threads = 20
        for root, dirs, files in os.walk(self.report_path):
            for mfile in files:
                while len(thlist) >= max_threads:
                    time.sleep(5)
                    for t in thlist:
                        t.join(2.0)
                        if not t.isAlive():
                            thlist.remove(t)
                            alogger.info("finish convert one report")
                full_path = os.path.join(root, mfile)
                t = Thread(target=func, args=(full_path,))
                thlist.append(t)
                t.start()

        for t in thlist:
            t.join()
            alogger.info("finish convert one report")

    def deal_process(self, func):
        plist = []
        max_process = 20
        for root, dirs, files in os.walk(self.report_path):
            for mfile in files:
                while len(plist) >= max_process:
                    for p in plist:
                        p.join(1.0)
                        if not p.is_alive():
                            plist.remove(p)
                            alogger.info("finish convert one report")
                full_path = os.path.join(root, mfile)
                p = Process(target=func, args=(full_path,))
                plist.append(p)
                p.start()

        for p in plist:
            p.join()

    def cuckoo2txt(self, mode="process"):
        if mode == "process":
            self.deal_process(self.convert)
        else:
            self.deal_thread(self.convert)

