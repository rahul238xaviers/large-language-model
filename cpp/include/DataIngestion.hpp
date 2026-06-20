#ifndef DATA_INGESTION_HPP
#define DATA_INGESTION_HPP

#include <string>
#include <vector>

class DataIngestion {
public:
    DataIngestion(const std::string& data_dir);
    bool load_shard(const std::string& shard_name);
    
private:
    std::string data_dir_;
};

#endif // DATA_INGESTION_HPP
