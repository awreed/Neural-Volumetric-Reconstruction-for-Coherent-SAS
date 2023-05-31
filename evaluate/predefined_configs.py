import numpy as np


def get_expnames_pfa():
    expnames_total_dict = {}
    
    cube_expnames = []
    cube_expnames.append("cube_20k_pfa")
    
    cylinder_expnames = []
    cylinder_expnames.append("cylinder_20k_pfa")

    bunny_expnames = []
    bunny_expnames.append("bunny_20k_pfa")

    armadilo_expnames = []
    armadilo_expnames.append("armadilo_20k_pfa")

    buddha_expnames = []
    buddha_expnames.append("buddha_20k_pfa")
    
    lucy_expnames = []
    lucy_expnames.append("lucy_20k_pfa")

    dragon_expnames = []
    dragon_expnames.append("dragon_20k_pfa")

    xyz_dragon_expnames = []
    xyz_dragon_expnames.append("xyz_dragon_20k_pfa")


    expnames_total_dict["lucy"] = lucy_expnames
    expnames_total_dict["dragon"] = dragon_expnames
    expnames_total_dict["xyz_dragon"] = xyz_dragon_expnames
    expnames_total_dict["buddha"] = buddha_expnames
    expnames_total_dict["cube"] = cube_expnames
    expnames_total_dict["cylinder"] = cylinder_expnames
    expnames_total_dict["bunny"] = bunny_expnames
    expnames_total_dict["armadilo"] = armadilo_expnames
    
    return expnames_total_dict


def get_expnames():
    expnames_total_dict = {}

    bunny_expnames = []
    for k in [20]:
        for n in [0, 10, 20]:
            bunny_expnames.append("bunny_%dk_%ddb" % (k, n))
            bunny_expnames.append("bunny_%dk_%ddb_no_network" % (k, n))
            bunny_expnames.append("bunny_%dk_%ddb_das" % (k, n))

    armadilo_expnames = []
    for k in [20]:
        for n in ['m20', 'm10', '0', '10', '20']:
            armadilo_expnames.append("armadilo_%dk_%sdb" % (k, n))
            armadilo_expnames.append("armadilo_%dk_%sdb_no_network" % (k, n))
            armadilo_expnames.append("armadilo_%dk_%sdb_das" % (k, n))

    buddha_expnames = []
    buddha_expnames.append("buddha_20k_20db")
    buddha_expnames.append("buddha_20k_20db_no_network")
    buddha_expnames.append("buddha_20k_20db_das")
    
    lucy_expnames = []
    lucy_expnames.append("lucy_20k_20db")
    lucy_expnames.append("lucy_20k_20db_no_network")
    lucy_expnames.append("lucy_20k_20db_das")

    dragon_expnames = []
    dragon_expnames.append("dragon_20k_20db")
    dragon_expnames.append("dragon_20k_20db_no_network")
    dragon_expnames.append("dragon_20k_20db_das")

    xyz_dragon_expnames = []
    xyz_dragon_expnames.append("xyz_dragon_20k_20db")
    xyz_dragon_expnames.append("xyz_dragon_20k_20db_no_network")
    xyz_dragon_expnames.append("xyz_dragon_20k_20db_das")

    cube_expnames = []
    cube_expnames.append("cube_20k_20db")
    cube_expnames.append("cube_20k_20db_no_network")
    cube_expnames.append("cube_20k_20db_das")

    cylinder_expnames = []
    cylinder_expnames.append("cylinder_20k_20db")
    cylinder_expnames.append("cylinder_20k_20db_no_network")
    cylinder_expnames.append("cylinder_20k_20db_das")

    expnames_total_dict["bunny"] = bunny_expnames
    expnames_total_dict["armadilo"] = armadilo_expnames
    expnames_total_dict["lucy"] = lucy_expnames
    expnames_total_dict["dragon"] = dragon_expnames
    expnames_total_dict["cube"] = cube_expnames
    expnames_total_dict["cylinder"] = cylinder_expnames
    expnames_total_dict["xyz_dragon"] = xyz_dragon_expnames
    expnames_total_dict["buddha"] = buddha_expnames
    
    return expnames_total_dict

def get_camera_pose_dict():
    camera_poses = {}

    camera_poses["default"] = {
        'elevation': 0.1,
        'distance': 0.3,
        'fov': np.pi / 3.0
    }

    camera_poses["xyz_dragon"] = {
        'elevation': 0.1,
        'distance': 0.35,
        'fov': np.pi / 3.0
    }

    camera_poses["lucy"] = {
        'elevation': 0.1,
        'distance': 0.25,
        'fov': np.pi / 3.0
    }

    camera_poses["cube"] = {
        'elevation': 0.21,
        'elevation_to': 0.07,
        'distance': 0.45,
        'fov': np.pi / 4.0
    }

    camera_poses["cylinder"] = {
        'elevation': 0.1,
        'elevation_to': 0.1,
        'distance': 0.45,
        'fov': np.pi / 4.0
    }

    return camera_poses