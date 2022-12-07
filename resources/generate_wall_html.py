import os
import shutil
for f in os.listdir('/Users/jinlinyi/Documents/cvpr2023_perspectivefield_arXiv/fig/wild/merged'):
    if f == '.DS_Store':
        continue
    print(f"""
    <tr>
      <td width=47%>
          <div>
              <p align="center">
                  <img src="resources/merged/{f}/rgb.jpg" width="100%" style="border:1px solid black;"></img>
              </p>
          </div>
      </td>
      <td width=6%>
          <center>
              <!-- <h3> </h3> -->
          </center>
      </td>
      <td width=47%>
          <div>
              <p align="center">
                <img src="resources/merged/{f}/perspective_pred.jpg" width="100%" style="border:1px solid black;"></img>
              </p>
          </div>
      </td>
    </tr>
    """)
    os.makedirs(f'merged/{f}', exist_ok=True)
    shutil.copyfile(
        f'/Users/jinlinyi/Documents/cvpr2023_perspectivefield/fig/wild/merged/{f}/_rgb.jpg',
        f'merged/{f}/rgb.jpg',
    )
    shutil.copyfile(
        f'/Users/jinlinyi/Documents/cvpr2023_perspectivefield/fig/wild/merged/{f}/perspective_pred.jpg',
        f'merged/{f}/perspective_pred.jpg',
    )