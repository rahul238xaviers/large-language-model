#include "DataIngestion.hpp"
#include <iostream>

DataIngestion::DataIngestion(const std::string& data_dir) : data_dir_(data_dir) {
    std::cout << "DataIngestion initialized with directory: " << data_dir_ << std::endl;
}

bool DataIngestion::load_shard(const std::string& shard_name) {
    std::cout << "Loading shard: " << shard_name << " from " << data_dir_ << std::endl;
    return true;
}
