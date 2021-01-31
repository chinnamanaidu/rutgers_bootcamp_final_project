/* =============================================================
 * bootstrap-typeahead.js v2.0.0
 * http://twitter.github.com/bootstrap/javascript.html#typeahead
 * =============================================================
 * Copyright 2012 Twitter, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================ */

!function( $ ){

  "use strict"

  var Typeahead = function ( element, options ) {
    this.$element = $(element)
    this.options = $.extend({}, $.fn.typeahead.defaults, options)
    this.matcher = this.options.matcher || this.matcher
    this.sorter = this.options.sorter || this.sorter
    this.highlighter = this.options.highlighter || this.highlighter
    this.$menu = $(this.options.menu).appendTo('body')
    this.source = this.options.source
    this.shown = false
    this.listen()
  }

  Typeahead.prototype = {

    constructor: Typeahead

  , select: function () {
      var val = this.$menu.find('.active').attr('data-value')
      this.$element.val(val)
      return this.hide()
    }

  , show: function () {
      var pos = $.extend({}, this.$element.offset(), {
        height: this.$element[0].offsetHeight
      })

      this.$menu.css({
        top: pos.top + pos.height
      , left: pos.left
      })

      this.$menu.show()
      this.shown = true
      return this
    }

  , hide: function () {
      this.$menu.hide()
      this.shown = false
      return this
    }

  , lookup: function (event) {
      var that = this
        , items
        , q

      this.query = this.$element.val()

      if (!this.query) {
        return this.shown ? this.hide() : this
      }

      items = $.grep(this.source, function (item) {
        if (that.matcher(item)) return item
      })

      items = this.sorter(items)

      if (!items.length) {
        return this.shown ? this.hide() : this
      }

      return this.render(items.slice(0, this.options.items)).show()
    }

  , matcher: function (item) {
      return ~item.toLowerCase().indexOf(this.query.toLowerCase())
    }

  , sorter: function (items) {
      var beginswith = []
        , caseSensitive = []
        , caseInsensitive = []
        , item

      while (item = items.shift()) {
        if (!item.toLowerCase().indexOf(this.query.toLowerCase())) beginswith.push(item)
        else if (~item.indexOf(this.query)) caseSensitive.push(item)
        else caseInsensitive.push(item)
      }

      return beginswith.concat(caseSensitive, caseInsensitive)
    }

  , highlighter: function (item) {
      return item.replace(new RegExp('(' + this.query + ')', 'ig'), function ($1, match) {
        return '<strong>' + match + '</strong>'
      })
    }

  , render: function (items) {
      var that = this

      items = $(items).map(function (i, item) {
        i = $(that.options.item).attr('data-value', item)
        i.find('a').html(that.highlighter(item))
        return i[0]
      })

      items.first().addClass('active')
      this.$menu.html(items)
      return this
    }

  , next: function (event) {
      var active = this.$menu.find('.active').removeClass('active')
        , next = active.next()

      if (!next.length) {
        next = $(this.$menu.find('li')[0])
      }

      next.addClass('active')
    }

  , prev: function (event) {
      var active = this.$menu.find('.active').removeClass('active')
        , prev = active.prev()

      if (!prev.length) {
        prev = this.$menu.find('li').last()
      }

      prev.addClass('active')
    }

  , listen: function () {
      this.$element
        .on('blur',     $.proxy(this.blur, this))
        .on('keypress', $.proxy(this.keypress, this))
        .on('keyup',    $.proxy(this.keyup, this))

      

      this.$menu
        .on('click', $.proxy(this.click, this))
		.on('select', $.proxy(this.click, this))
        .on('mouseenter', 'li', $.proxy(this.mouseenter, this))
    }

  , keyup: function (e) {
      e.stopPropagation()
      e.preventDefault()

      switch(e.keyCode) {
        case 40: // down arrow
        case 38: // up arrow
          break

        case 9: // tab
        case 13: // enter
          if (!this.shown) return
          this.select()
          break

        case 27: // escape
          this.hide()
          break

        default:
          this.lookup()
      }

  }

  , keypress: function (e) {
      e.stopPropagation()
      if (!this.shown) return

      switch(e.keyCode) {
        case 9: // tab
        case 13: // enter
        case 27: // escape
          e.preventDefault()
          break

        case 38: // up arrow
          e.preventDefault()
          this.prev()
          break

        case 40: // down arrow
          e.preventDefault()
          this.next()
          break
      }
    }

  , blur: function (e) {
      var that = this
      e.stopPropagation()
      e.preventDefault()
      setTimeout(function () { that.hide() }, 150)
    }

  , click: function (e) {
      e.stopPropagation()
      e.preventDefault()
      this.select()
    }

  , mouseenter: function (e) {
      this.$menu.find('.active').removeClass('active')
      $(e.currentTarget).addClass('active')
    }

  }


  /* TYPEAHEAD PLUGIN DEFINITION
   * =========================== */

  $.fn.typeahead = function ( option ) {
    return this.each(function () {
      var $this = $(this)
        , data = $this.data('typeahead')
        , options = typeof option == 'object' && option
      if (!data) $this.data('typeahead', (data = new Typeahead(this, options)))
      if (typeof option == 'string') data[option]()
    })
  }

  $.fn.typeahead.defaults = {
    source: ['IPOE',
'TSLA',
'CCIV',
'INO',
'RIG',
'AG',
'FB',
'FUBO',
'TLRY',
'SPCE',
'NNDM',
'CMCSA',
'C',
'BBD',
'ABEV',
'VZ',
'SRNE',
'NCLH',
'SWN',
'ERIC',
'GOLD',
'PLUG',
'M',
'ACB',
'IPOE',
'TSLA',
'CCIV',
'INO',
'RIG',
'AG',
'FB',
'FUBO',
'TLRY',
'SPCE',
'NNDM',
'CMCSA',
'C',
'BBD',
'ABEV',
'VZ',
'SRNE',
'NCLH',
'SWN',
'ERIC',
'GOLD',
'PLUG',
'M',
'ACB',
'NLY',
'GM',
'UAL',
'VIAC',
'WKHS',
'X',
'ET',
'AUY',
'RKT',
'V',
'MRO',
'KMI',
'VALE',
'BBBY',
'OXY',
'JNJ',
'IFF',
'PBR',
'APHA',
'BSX',
'KGC',
'UBER',
'RYCEY',
'PBCT',
'HL',
'FCX',
'DAL',
'MU',
'SCHW',
'TWTR',
'CVE',
'KR',
'CSCO',
'CLNY',
'ZNGA',
'HBAN',
'DISCA',
'DKNG',
'IBN',
'DISCK',
'RTX',
'SLB',
'WDC',
'CLF',
'NVT',
'LUMN',
'SNAP',
'AMCR',
'JPM',
'BABA',
'HPQ',
'MS',
'TSM',
'MDLZ',
'MAC',
'NKLA',
'BMY',
'KO',
'XPEV',
'CX',
'RLX',
'BA',
'COTY',
'LI',
'JNPR',
'CLOV',
'AZN',
'SU',
'SPWR',
'BP',
'IBM',
'GILD',
'COP',
'AMRN',
'PSLV',
'CVX',
'ORCL',
'APA',
'HAL',
'JMIA',
'BFT',
'RF',
'PK',
'ABT',
'EBAY',
'WMT',
'DIS',
'TWNK',
'PINS',
'CGC',
'BEN',
'WBA',
'LVS',
'PCG',
'OTIS',
'UMC',
'TEVA',
'DVN',
'PYPL',
'GSX',
'JD',
'MO',
'KIM',
'SQ',
'ABBV',
'SYF',
'WMB',
'BILI',
'SWKS',
'TME',
'JBLU',
'DOW',
'AGNC',
'LUV',
'OPK',
'GFI',
'FLEX',
'INFY',
'AM',
'NEE',
'PG',
'VTRS',
'SBUX',
'BTG',
'SABR',
'HST',
'BLNK',
'HRL',
'IPOF',
'LKNCY',
'WORK',
'QS',
'ON',
'AMAT',
'MRVL',
'WY',
'HBI',
'IRTC',
'FUTU',
'EPD',
'XL',
'CL',
'BK',
'MRK',
'USB',
'QRTEA',
'HPE',
'CDE',
'AES',
'HMY',
'MGM',
'NKE',
'FSR',
'LLY',
'GIS',
'QCOM',
'BKR',
'FE',
'ATVI',
'LYG',
'RRC',
'IRM',
'TFC',
'DDD',
'SLM',
'ATUS',
'COG',
'CARR',
'MA',
'PEP',
'CNK',
'SAN',
'NEM',
'BNTX',
'GPS',
'ED',
'JWN',
'IQ',
'CFG',
'CRM',
'CVS',
'LB',
'PAAS',
'IPG',
'BRK-B',
'KEY',
'MT',
'FTI',
'FHN',
'RIDE',
'PRSP',
'RMO',
'MPC',
'AA',
'EQT',
'NVDA',
'BCS',
'BIDU',
'CCJ',
'TIGR',
'JCI',
'NTAP',
'AEO',
'TXN',
'BAM',
'NLOK',
'LU',
'VIPS',
'PPL',
'ETRN',
'HWM',
'PBR-A',
'CLNE',
'GGB',
'EOG',
'LAC',
'DHR',
'DCT',
'PSX',
'K',
'DBX',
'UNM',
'MDT',
'LYFT',
'SKLZ',
'OPEN',
'VOD',
'WU',
'GLW',
'FISV',
'FTCH',
'EQH',
'NOV',
'CRON',
'RCL',
'SAVE',
'BLDP',
'AFL',
'DHI',
'MET',
'NI',
'UMPQ',
'GOEV',
'GSK',
'PAA',
'AXP',
'NRZ']
  , items: 8
  , menu: '<ul class="typeahead dropdown-menu"></ul>'
  , item: '<li><a href="https://www.yahoo.com">https://www.yahoo.com</a></li>'
  }

  $.fn.typeahead.Constructor = Typeahead


 /* TYPEAHEAD DATA-API
  * ================== */

  $(function () {
    $('body').on('focus.typeahead.data-api', '[data-provide="typeahead"]', function (e) {
      var $this = $(this)
      if ($this.data('typeahead')) return
      e.preventDefault()
      $this.typeahead($this.data())
    })
  })

}( window.jQuery )