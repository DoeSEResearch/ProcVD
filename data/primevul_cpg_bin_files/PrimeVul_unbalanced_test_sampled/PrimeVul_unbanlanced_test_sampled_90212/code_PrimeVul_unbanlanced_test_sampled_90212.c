  virtual bool ethernet_connected() const {
    return ethernet_ ? ethernet_->connected() : false;
  }
