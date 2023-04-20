# Define the column types by the cleaning required.
# The union of these lists will be the set of columns processed from the original event data frame (~150 columns vs. 350).

bitcoin_cols = ['event.eventLabel',
    'request.eventLabel']

country_cols = ['event.metadata.addressCountry',
               'request.metadata.addressCountry']

airport_cols = ['event.metadata.authResponseEIN.headers.map.cf-ray']

#-------------
# Seperate column names and the 'n' values as list then zip into a dictionary.
first_n_cols = ['event.metadata.authResponseEIN.body.message',
    'event.metadata.authResponseEIN.bodyText',
    'event.metadata.blockioResponse.data.error_message',
    'event.metadata.prossessorResponse.error.fieldErrors',
    'request.metadata.processorResponse']

first_n_cols_n_values = [20, 35, 40, 30, 20]

first_n_char_cols_dict = dict( zip(first_n_cols, first_n_cols_n_values) )
# Old.
# {'event.metadata.authResponseEIN.body.message': 20,
#     'event.metadata.authResponseEIN.bodyText': 35,
#     'event.metadata.blockioResponse.data.error_message': 40,
#     'event.metadata.prossessorResponse.error.fieldErrors': 30,
#     'request.metadata.processorResponse': 20}
#-------------

url_cols = ['event.metadata.prossessorResponse.acsURL',
            'event.metadata.prossessorResponse.acs_url']

card_type_cols = ['event.metadata.prossessorResponse.card.type',
                  'request.metadata.prossessorResponse.card.type']

card_year_cols = ['event.metadata.prossessorResponse.card.cardExpiry.year',
                  'event.metadata.prossessorResponse.card_expiry_year',
                  'request.metadata.prossessorResponse.card.cardExpiry.year']

datetime_cols = ['event.created',
                'request.created']

province_cols = ['event.metadata.addressProvince',
                 'event.metadata.prossessorResponse.billingDetails.province',
                 'event.metadata.prossessorResponse.billingDetails.state',
                 'event.metadata.province',
                 'request.metadata.addressProvince',
                 'request.metadata.prossessorResponse.billingDetails.province']

string_list_cols = ['event.metadata.authResponseEIN.headers.map.Content-Length',
    'event.metadata.authResponseEIN.headers.map.age',
    'event.metadata.authResponseEIN.headers.map.content-length',
    'event.metadata.authResponseEIN.headers.map.status']

string_to_lc_cols = ['event.metadata.addressCity',
                     'event.metadata.prossessorResponse.billingDetails.city',
                     'event.metadata.prossessorResponse.billingDetails.zip',
                     'request.metadata.prossessorResponse.billingDetails.city',
                     'request.metadata.addressCity']

float_cols = ['event.metadata.requestParams.price',
              'request.metadata.cents',
              'event.metadata.price',
              'event.metadata.cents',
              'event.metadata.firstAmount',
              'event.metadata.mongoResponse.amount',
              'event.metadata.mongoResponse.price',
              'event.metadata.rate',
              'event.metadata.requestParams.amount',
              'event.metadata.requestParams.product_amount',
              'event.metadata.secondAmount',
              'request.metadata.amount',
              'request.metadata.mongoResponse.amount',
              'request.metadata.mongoResponse.price',
              'request.metadata.price',
              'request.metadata.requestParams.amount',
              'request.metadata.requestParams.price',
              'request.metadata.requestParams.product_amount',
              'event.metadata.amount',
              'event.metadata.blockioResponse.data.amount_sent',
              'event.metadata.blockioResponse.data.amount_withdrawn',
              'event.metadata.blockioResponse.data.network_fee',
              'event.metadata.cashAdvanceReimbursement',
              'event.metadata.lastTradedPx',
              'event.metadata.prossessorResponse.amount',
              'event.metadata.prossessorResponse.charge_amount',
              'event.metadata.requestParams.charge_amount',
              'event.value',
              'request.metadata.cashAdvanceReimbursement',
              'request.metadata.rate',
              'request.metadata.requestParams.charge_amount',
              'request.value',
              'event.metadata.passwordLength']

# id_cols currently not used. Here for completeness.
# No cleaning required.
id_cols = ['event._id',
    'request._id',
    'request.metadata.email']

# Columns with no cleaning required.
no_cleaning_cols = ['event.metadata.userAgent',
    'request.metadata.userAgent',
    'event.eventAction',
    'event.eventCategory',
    'event.metadata.apTransferResponse.errorcode',
    'event.metadata.apTransferResponse.errormsg',
    'event.metadata.apTransferResponse.result',
    'event.metadata.authResponseAP.AuthType',
    'event.metadata.authResponseAP.Authenticated',
    'event.metadata.authResponseEIN.ok',
    'event.metadata.authResponseEIN.status',
    'event.metadata.authResponseEIN.statusText',
    'event.metadata.blockioResponse.data.estimated_network_fee',
    'event.metadata.blockioResponse.data.network',
    'event.metadata.blockioResponse.status',
    'event.metadata.city',
    'event.metadata.instrument',
    'event.metadata.mongoResponse.product',
    'event.metadata.paymentMethod',
    'event.metadata.product',
    'event.metadata.productId',
    'event.metadata.prossessorResponse.authentication.eci',
    'event.metadata.prossessorResponse.authentication.signatureStatus',
    'event.metadata.prossessorResponse.authentication.threeDEnrollment',
    'event.metadata.prossessorResponse.authentication.threeDResult',
    'event.metadata.prossessorResponse.avsResponse',
    'event.metadata.prossessorResponse.billingDetails.country',
    'event.metadata.prossessorResponse.card.cardExpiry.month',
    'event.metadata.prossessorResponse.card.cardType',
    'event.metadata.prossessorResponse.card_expiry_month',
    'event.metadata.prossessorResponse.card_type',
    'event.metadata.prossessorResponse.cvvVerification',
    'event.metadata.prossessorResponse.error.code',
    'event.metadata.prossessorResponse.error.links',
    'event.metadata.prossessorResponse.error.message',
    'event.metadata.prossessorResponse.links',
    'event.metadata.prossessorResponse.merchantDescriptor.dynamicDescriptor',
    'event.metadata.prossessorResponse.riskReasonCode',
    'event.metadata.prossessorResponse.signatureStatus',
    'event.metadata.prossessorResponse.status',
    'event.metadata.prossessorResponse.threeDEnrollment',
    'event.metadata.prossessorResponse.threeDResult',
    'event.metadata.requestParams.enrollment_status',
    'event.metadata.requestParams.product',
    'event.metadata.requestParams.redirectUrl',
    'event.metadata.requestParams.redirect_url',
    'event.metadata.requestParams.threeDEnrollment',
    'event.metadata.side',
    'event.metadata.status',
    'event.metadata.successUrl',
    'event.metadata.tradesResponse',
    'event.metadata.twoFactorResponse.Authenticated',
    'event.metadata.type',
    'request.eventAction',
    'request.eventCategory',
    'request.metadata.failureUrl',
    'request.metadata.mongoResponse.product',
    'request.metadata.mongoResponse.status',
    'request.metadata.product',
    'request.metadata.prossessorResponse.avsResponse',
    'request.metadata.prossessorResponse.billingDetails.country',
    'request.metadata.prossessorResponse.card.cardExpiry.month',
    'request.metadata.prossessorResponse.status',
    'request.metadata.requestParams.enrollment_status',
    'request.metadata.requestParams.product',
    'request.metadata.requestParams.redirectUrl',
    'request.metadata.requestParams.redirect_url',
    'request.metadata.requestParams.threeDEnrollment',
    'request.metadata.status',
    'request.metadata.successUrl',
    'request.metadata.type',
    'fraud']
