  virtual NetworkIPConfigVector GetIPConfigs(const std::string& device_path,
                                             std::string* hardware_address) {
    hardware_address->clear();
    NetworkIPConfigVector ipconfig_vector;
    if (EnsureCrosLoaded() && !device_path.empty()) {
      IPConfigStatus* ipconfig_status = ListIPConfigs(device_path.c_str());
      if (ipconfig_status) {
        for (int i = 0; i < ipconfig_status->size; i++) {
          IPConfig ipconfig = ipconfig_status->ips[i];
          ipconfig_vector.push_back(
              NetworkIPConfig(device_path, ipconfig.type, ipconfig.address,
                              ipconfig.netmask, ipconfig.gateway,
                              ipconfig.name_servers));
        }
        *hardware_address = ipconfig_status->hardware_address;
        FreeIPConfigStatus(ipconfig_status);
        std::sort(ipconfig_vector.begin(), ipconfig_vector.end());
      }
    }
    return ipconfig_vector;
  }
