from setuptools import find_packages, setup

package_name = 'ros2_onnxruntime_imgclassifier'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ertugrul',
    maintainer_email='ertugrulsalihileri@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'onnxruntime_classifier_node = ros2_onnxruntime_imgclassifier.onnxruntime_classifier_node:main',
            'image_publisher = ros2_onnxruntime_imgclassifier.image_publisher:main',
        ],
    },
)
