import component as cn
import os

if __name__ == '__main__':

    params = cn.read_params()
    data_loader, algorithm = cn.initialize(params)
    algorithm.save_folder = data_loader.nickname + '/C' + str(params['C']) + '/' + params['module'] + '/' + params['algorithm'] + '/'
    if not os.path.exists(algorithm.save_folder):
        os.makedirs(algorithm.save_folder)
    algorithm.save_name = 'seed' + str(params['seed']) + ' N' + str(data_loader.pool_size) + ' C' + str(
        params['C']) + ' R' + str(params['R']) + ' ' + algorithm.save_name
    algorithm.run()
