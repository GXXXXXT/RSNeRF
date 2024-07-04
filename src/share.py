from multiprocessing.managers import BaseManager, NamespaceProxy
from copy import deepcopy
import torch.multiprocessing as mp
from time import sleep
import sys

class ShareDataProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__')

class ShareData:
    global lock
    lock = mp.RLock()

    def __init__(self):
        self.__stop_mapping = False
        self.__stop_tracking = False

        self.__exp_decoder = None
        self.__road_decoder = None
        self.__voxels = None
        self.__octree = None
        self.__states = None
        self.__psnr = {}
        self.__tracking_trajectory = []

    @property
    def exp_decoder(self):
        with lock:
            return deepcopy(self.__exp_decoder)
            print("========== exp_decoder get ==========")
            sys.stdout.flush()

    @exp_decoder.setter
    def exp_decoder(self, exp_decoder):
        with lock:
            self.__exp_decoder = deepcopy(exp_decoder)
            # print("========== exp_decoder set ==========")
            sys.stdout.flush()

    @property
    def road_decoder(self):
        with lock:
            return deepcopy(self.__road_decoder)
            print("========== road_decoder get ==========")
            sys.stdout.flush()
    
    @road_decoder.setter
    def road_decoder(self, road_decoder):
        with lock:
            self.__road_decoder = deepcopy(road_decoder)
            # print("========== road_decoder set ==========")
            sys.stdout.flush()

    @property
    def voxels(self):
        with lock:
            return deepcopy(self.__voxels)
            print("========== voxels get ==========")
            sys.stdout.flush()
    
    @voxels.setter
    def voxels(self, voxels):
        with lock:
            self.__voxels = deepcopy(voxels)
            print("========== voxels set ==========")
            sys.stdout.flush()

    @property
    def octree(self):
        with lock:
            return deepcopy(self.__octree)
            print("========== octree get ==========")
            sys.stdout.flush()
    
    @octree.setter
    def octree(self, octree):
        with lock:
            self.__octree = deepcopy(octree)
            print("========== octree set ==========")
            sys.stdout.flush()

    @property
    def states(self):
        with lock:
            return deepcopy(self.__states)
            print("========== states get ==========")
            sys.stdout.flush()
    
    @states.setter
    def states(self, states):
        with lock:
            self.__states = deepcopy(states)
            # print("========== states set ==========")
            sys.stdout.flush()

    @property
    def stop_mapping(self):
        with lock:
            return self.__stop_mapping
            print("========== stop_mapping get ==========")
            sys.stdout.flush()
    
    @stop_mapping.setter
    def stop_mapping(self, stop_mapping):
        with lock:
           self.__stop_mapping = stop_mapping
           print("========== stop_mapping set ==========")
           sys.stdout.flush()

    @property
    def stop_tracking(self):
        with lock:
            return self.__stop_tracking
            print("========== stop_tracking get ==========")
            sys.stdout.flush()
    
    @stop_tracking.setter
    def stop_tracking(self, stop_tracking):
        with lock:
           self.__stop_tracking = stop_tracking
           print("========== stop_tracking set ==========")
           sys.stdout.flush()

    @property
    def tracking_trajectory(self):
        with lock:
            return deepcopy(self.__tracking_trajectory)
            print("========== tracking_trajectory get ==========")
            sys.stdout.flush()

    def push_pose(self, pose):
        with lock:
            self.__tracking_trajectory.append(deepcopy(pose))
            # print("========== push_pose ==========")
            sys.stdout.flush()

    @property
    def psnr(self):
        with lock:
            return deepcopy(self.__psnr)
            print("========== psnr get ==========")
            sys.stdout.flush()

    @psnr.setter
    def psnr(self, psnr):
        with lock:
            self.__psnr = psnr
            print("========== psnr set ==========")
            sys.stdout.flush()

    def push_psnr(self, stamp, psnr):
        with lock:
            self.__psnr[f"frame_{stamp}"] = psnr
            # print("========== push_pose ==========")
            sys.stdout.flush()
