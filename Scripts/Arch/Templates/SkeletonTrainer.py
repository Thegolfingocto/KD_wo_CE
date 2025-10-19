'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


from ITrainer import ITrainer
from typing import List
import torch

class KDTrainer(ITrainer):
    """
    Example skeleton code to expidite future dev
    """
    def __init__(self, strCacheDir: str = "", dConfig: dict = {}, vecIgnoredFields: List[str] = [], bStartLog: bool = False):
        super().__init__(strCacheDir, dConfig, vecIgnoredFields, bStartLog)
        
    #Details of ID generation
    def ModifyHashCfg(self) -> None:
        """
        Modify HashCfg as desired
        """
        self.HashCfg = self.HashCfg
        
    #Print config util for DisplayCache
    def PrintConfigSummary(self, dTempCfg: dict) -> None:
        print("Add details from dTempCfg here")
        
    def LoadDataset(self) -> tuple[any, any]:
        """
        Load dataset here
        """
        return None, None
    
    def LoadModel(self) -> torch.nn.Module:
        """
        Load model here
        """
        return None
    
    def LoadOptimizer(self) -> any:
        return None
    
    def LoadLossFcn(self) -> any:
        return None
    
    def LoadScheduler(self) -> any:
        return None
    
    def LoadComponents(self) -> None:
        """
        Catchall funtion for loading any additional derived class specific stuff
        """
        return None