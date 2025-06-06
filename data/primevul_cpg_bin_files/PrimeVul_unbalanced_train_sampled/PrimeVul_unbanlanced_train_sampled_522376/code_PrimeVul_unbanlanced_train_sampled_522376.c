NAN_METHOD(TxnWrap::ctor) {
    Nan::HandleScope scope;

    EnvWrap *ew = Nan::ObjectWrap::Unwrap<EnvWrap>(Local<Object>::Cast(info[0]));
    int flags = 0;

    if (info[1]->IsObject()) {
        Local<Object> options = Local<Object>::Cast(info[1]);

        // Get flags from options

        setFlagFromValue(&flags, MDB_RDONLY, "readOnly", false, options);
    }
    
    // Check existence of current write transaction
    if (0 == (flags & MDB_RDONLY) && ew->currentWriteTxn != nullptr) {
        return Nan::ThrowError("You have already opened a write transaction in the current process, can't open a second one.");
    }

    MDB_txn *txn;
    int rc = mdb_txn_begin(ew->env, nullptr, flags, &txn);
    if (rc != 0) {
        if (rc == EINVAL) {
            return Nan::ThrowError("Invalid parameter, which on MacOS is often due to more transactions than available robust locked semaphors (see node-lmdb docs for more info)");
        }
        return throwLmdbError(rc);
    }

    TxnWrap* tw = new TxnWrap(ew->env, txn);
    tw->flags = flags;
    tw->ew = ew;
    tw->ew->Ref();
    tw->Wrap(info.This());
    
    // Set the current write transaction
    if (0 == (flags & MDB_RDONLY)) {
        ew->currentWriteTxn = tw;
    }
    else {
        ew->readTxns.push_back(tw);
    }

    return info.GetReturnValue().Set(info.This());
}