from setuptools import setup
from setuptools import find_packages

package_name = 'etri_msfloc'
packages = find_packages(exclude=['test'])

setup(
    name=package_name,
    version='0.0.0',
    packages=packages,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wicomai',
    maintainer_email='wicomai@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "test_run = etri_msfloc.test_node:main",
            "get_rgb_image = etri_msfloc.image_subscriber:main",
            "depth_predictor = etri_msfloc.depth_predictor:main",
            "camlid_calibration = etri_msfloc.cam_lidar_calibration:main",
            "pcd_projection = etri_msfloc.pcd_projection:main"
        ],
    },
)
