header('Content-type:text/html; charset=UTF-8;');

$action = (isset($_GET['action'])) ? $_GET['action'] : null;
$symbol = (isset($_GET['symbol'])) ? $_GET['symbol'] : null;

switch($action) {
    case 'autocjson':
        getYahooSymbolAutoComplete($symbol);
        break;
}

function getYahooSymbolAutoCompleteJson($symbolChar) {
    $data = @file_get_contents("http://d.yimg.com/aq/autoc?callback=YAHOO.util.ScriptNodeDataSource.callbacks&query=$symbolChar");

    // parse yahoo data into a list of symbols
    $result = [];
    $json = json_decode(substr($data, strlen('YAHOO.util.ScriptNodeDataSource.callbacks('), -1));

    foreach ($json->ResultSet->Result as $stock) {
        $result[] = '('.$stock->symbol.') '.$stock->name;
    }

    echo json_encode(['symbols' => $result]);
}