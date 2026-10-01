import numpy as np
d = np.load("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler/grasp_dataset/cube_5cm_grasps_valid.npz", allow_pickle=True)
g = d["grasp_pose"]
i = 12
pre, grasp, sq = g[i,0,0,7:], g[i,0,1,7:], g[i,0,2,7:]
names = ["th_yaw","th_pitch","index","middle","ring","pinky"]
print("%9s %9s %9s %9s %9s" % ("joint","pregrasp","grasp","squeeze","sq-grasp"))
for n,a,b,c in zip(names,pre,grasp,sq):
    print("%9s %9.3f %9.3f %9.3f %+9.3f" % (n,a,b,c,c-b))
print()
print("L2 |squeeze - grasp| = %.4f rad" % float(np.linalg.norm(sq-grasp)))
print("L2 |grasp - pregrasp| = %.4f rad" % float(np.linalg.norm(grasp-pre)))
