import logging, os, torch                                                                                                                              
import hydra                                                                                                                                           
from omegaconf import DictConfig                                                                                                                       
from lightning.pytorch import seed_everything                                                                                                          
from src.train.training import build_model_config                                                                                  
from src.model.model import PFAGCN                                                                                                                     
from src.modules.dataloader import build_manifest_dataloader, load_ia_weights                                                                          
from src.utils.system_runtime import apply_system_env                                                                                                  
                                                                                                                                                        
log = logging.getLogger(__name__)                                                                                                                      
                                                                                                                                                        
def _load_checkpoint(model, path, device):                                                                                                             
    state = torch.load(path, map_location=device)                                                                                                      
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=True)                                                        
    model.to(device).eval()                                                                                                                            
    return model                                                                                                                                       
                                                                                                                                                        
@hydra.main(config_path="../../configs", config_name="eval")                                                                                           
def main(cfg: DictConfig) -> None:                                                                                                                     
    apply_system_env(cfg)                                                                                                                              
    seed_everything(int(cfg.get("seed", 42)))                                                                                                          
    device = torch.device(cfg.get("device", "cpu"))                                                                                                    
    model = PFAGCN(build_model_config(cfg)).to(device)                                                                                                 
    model = _load_checkpoint(model, cfg.checkpoint_path, device)                                                                                       
                                                                                                                                                        
    loader = build_manifest_dataloader(                                                                                                                
        cfg.data.manifest, cfg.data, base_dir=hydra.utils.get_original_cwd(), shuffle=False                                                            
    )                                                                                                                                                  
    ia_weights = load_ia_weights(cfg, os.path(hydra.utils.get_original_cwd()))                                                                            
                                                                                                                        
                                                                                                                                                        
if __name__ == "__main__":                                                                                                                             
    main()