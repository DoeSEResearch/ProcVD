XMLTree::write() const
{
	xmlDocPtr doc;
	XMLNodeList children;
	int result;

	xmlKeepBlanksDefault(0);
	doc = xmlNewDoc(xml_version);
	xmlSetDocCompressMode(doc, _compression);
	writenode(doc, _root, doc->children, 1);
	result = xmlSaveFormatFileEnc(_filename.c_str(), doc, "UTF-8", 1);
#ifndef NDEBUG
	if (result == -1) {
		xmlErrorPtr xerr = xmlGetLastError ();
		if (!xerr) {
			std::cerr << "unknown XML error during xmlSaveFormatFileEnc()." << std::endl;
		} else {
			std::cerr << "xmlSaveFormatFileEnc: error"
				<< " domain: " << xerr->domain
				<< " code: " << xerr->code
				<< " msg: " << xerr->message
				<< std::endl;
		}
	}
#endif
	xmlFreeDoc(doc);

	if (result == -1) {
		return false;
	}

	return true;
}