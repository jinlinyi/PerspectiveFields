
# Model Zoo

NOTE: Extract model weights under: `perspectiveField/models`

| Model Name and Weights                                                                                                    | Training Dataset                                                                                                          | Config File                                  | Outputs                                                           |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| [PersNet-360Cities](https://www.dropbox.com/s/czqrepqe7x70b7y/cvpr2023.pth)                                               | [360cities](https://www.360cities.net)                                                                                    | [View](../models/cvpr2023.yaml)                       | Perspective Field                                                 |
| [PersNet_paramnet-GSV-centered](https://www.dropbox.com/s/g6xwbgnkggapyeu/paramnet_gsv_rpf.pth)                           | [GSV](https://research.google/pubs/pub36899/)                                                                             | [View](../models/paramnet_gsv_rpf.yaml)               | Perspective Field + camera parameters (roll, pitch, vfov)         |
| [PersNet_Paramnet-GSV-uncentered](https://www.dropbox.com/s/ufdadxigewakzlz/paramnet_gsv_rpfpp.pth)                       | [GSV](https://research.google/pubs/pub36899/)                                                                             | [View](../models/paramnet_gsv_rpfpp.yaml)             | Perspective Field + camera parameters (roll, pitch, vfov, cx, cy) |
| [NEW:Paramnet-360Cities-edina-centered](https://www.dropbox.com/s/z2dja70bgy007su/paramnet_360cities_edina_rpf.pth)       | [360cities](https://www.360cities.net/) and [EDINA](https://github.com/tien-d/EgoDepthNormal/blob/main/README_dataset.md) | [View](../models/paramnet_360cities_edina_rpf.yaml)   | Perspective Field + camera parameters (roll, pitch, vfov)         |
| [NEW:Paramnet-360Cities-edina-uncentered](https://www.dropbox.com/s/nt29e1pi83mm1va/paramnet_360cities_edina_rpfpp.pth)  | [360cities](https://www.360cities.net/) and [EDINA](https://github.com/tien-d/EgoDepthNormal/blob/main/README_dataset.md) | [View](../models/paramnet_360cities_edina_rpfpp.yaml) | Perspective Field + camera parameters (roll, pitch, vfov, cx, cy) |

## Additional Comments

| Model Name                                        | Comments                                                                                     |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| PersNet-360Cities                                 | Trained on perspective images cropped from indoor, natural, and street view panoramas.       |
| PersNet_paramnet-GSV-centered                     | Trained on Google Street View data, assumes a centered principle point.                      |
| PersNet_Paramnet-GSV-uncentered                    | Trained on Google Street View data, makes no assumption about principle point location.      |
| NEW:Paramnet-360Cities-edina-centered              | Training data consists of a diverse set of indoor, outdoor, natural, and egocentric data. Trained on 360Cities and EDINA data, assumes a centered principle point. |
| NEW:Paramnet-360Cities-edina-uncentered            | Training data consists of a diverse set of indoor, outdoor, natural, and egocentric data. Trained on 360Cities and EDINA data, makes no assumption about principle point location. |
