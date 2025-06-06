  template<typename T> T GetWirelessNetworkByPath(
      std::vector<T>& networks, const std::string& path) {
    typedef typename std::vector<T>::iterator iter_t;
    iter_t iter = std::find_if(networks.begin(), networks.end(),
                               WirelessNetwork::ServicePathEq(path));
    return (iter != networks.end()) ? *iter : NULL;
  }
