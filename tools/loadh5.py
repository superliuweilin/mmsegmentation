import h5py

def explore_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as h5_file:
            print(f"Datasets in the .h5 file '{file_path}':")
            for dataset_name in h5_file.keys():
                print(f" -- {dataset_name}")
                # print(h5_file['Satellite_Source_Data'])

                # for group_name in group.keys():
                #     print(group_name)
                group = h5_file['Satellite_Source_Data']
            for member in group:
                member_data = group[member]
                print(member_data)

            print(f"\nAttributes in the .h5 file '{file_path}':")
            for attr_name, attr_value in h5_file.attrs.items():
                print(f" - {attr_name}: {attr_value}")


    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

if __name__ == "__main__":
    # 请将下面的文件路径替换为您要查看的.h5文件的路径
    file_path = "/home/lyu/lwl_wsp/mmsegmentation/data/LSDSSIMR/2020/20200301/20200301041500.h5"

    # 查看.h5文件中的数据集和文本数据
    explore_h5_file(file_path)
