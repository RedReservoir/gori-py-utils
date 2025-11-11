import xml
import xml.etree
import xml.etree.ElementTree



def save_xml(
    xml_obj,
    xml_filename
):
    """
    Saves an object to an XML file.

    Args:

        xml_obj (xml.etree.ElementTree):
            XML object to save.

        xml_filename (str):
            Name of the XML file.
    """

    def _pretty_print(current, parent=None, index=-1, depth=0):
        for i, node in enumerate(current):
            _pretty_print(node, current, i, depth + 1)
        if parent is not None:
            if index == 0:
                parent.text = '\n' + ('\t' * depth)
            else:
                parent[index - 1].tail = '\n' + ('\t' * depth)
            if index == len(parent) - 1:
                current.tail = '\n' + ('\t' * (depth - 1))

    #

    xml_file = open(xml_filename, "wb")

    xml_file.write(b"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n")

    _pretty_print(xml_obj.getroot())
    tree = xml.etree.ElementTree.ElementTree(xml_obj.getroot())
    tree.write(xml_file)

    xml_file.close()



def load_xml(
    xml_filename
):
    """
    Loads an object from an XML file.

    Args:

        xml_filename (str):
            Name of the XML file.

    Returns:
    
        xml.etree.ElementTree:
            Loaded XML object.
    """

    xml_obj = xml.etree.ElementTree.parse(xml_filename)

    return xml_obj
