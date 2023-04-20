import pandas as pd
import numpy as np

def clean_dataframe( df ):

# CLEAN THE DATAFRAME OVERALL BEFORE WORKING ON INDIVIDUAL COLUMNS.
# =================================================================
# CONVERT COLUMNS OF LIST TYPES TO STRING TO CATEGORIZE AFTER.
    # This was creating errors in running .nunique() on the table. We would get errors like:
    #   TypeError: ("unhashable type: 'list'", 'occurred at index event.metadata.authResponseEIN.headers.map.')
    #    
    list_columns = []

    for c in list( df.columns ):
        try:
            df[c].nunique()
        except TypeError:
            list_columns.append(c)    
            df[c] = df[c].apply( lambda x: str(x) ).replace(['nan'],np.nan)
            
    return df

# # REMOVE COLUMNS THAT ARE ALL NAN'S.
#     nan_columns = df.columns[ df.isna().all() ].tolist()

#     df.drop( labels=nan_columns, axis=1, inplace=True ) 

#     df = remove_columns( df )

# # WORK ON INDIVIDUAL COLUMNS.
# # ================================================================= 
# # Rename the older 'bitcoin' events to BTC so they match
#     df.loc[ df['event.eventLabel'].str.lower() == 'bitcoin', 'event.eventLabel' ] = 'BTC'
#     df.loc[ df['request.eventLabel'].str.lower() == 'bitcoin', 'request.eventLabel' ] = 'BTC'

# # # CONVERT APPROPRIATE COLUMNS TO DATETIME.
# #     # Do string conversions ....
#     df['event.created'] = pd.to_datetime(df['event.created']) # Seems already is datetime. Force it anyway.
#     df['request.created'] = pd.to_datetime(df['request.created']) # Already is datetime.
       
# # # Do timestamp conversions .... Not used right now.
# #     ## NOTE NAN'S ARE CONVERTED TO NAT'S. 
# #     df['event.metadata.updated'] = pd.to_datetime(df['event.metadata.updated']) # Already is datetime.
# #     df['request.metadata.updated'] = pd.to_datetime(df['request.metadata.updated']) # Already is datetime.

#     return df, list_columns, nan_columns

# def remove_columns( df ):
#     # NOTE: A number of the categorical variables have to have single (double/small?) occurances dropped.

#     columns_to_keep_list = [
#     #    'event.__v',   # All zeros.
#         'event._id',
#         'event.created',    # To DATETIME.
#         'event.eventAction', # CATEGORICAL.
#         'event.eventCategory', # CATEGORICAL.
#         'event.eventLabel',                ## BTC, ETH, etc. -> relationship.  # CATEGORICAL.
#         'event.metadata.addressCity',        ## To lower case, remove spaces.  # CATEGORICAL.
#         'event.metadata.addressCountry',     ## Convert to CA/US.   # CATEGORICAL.
#     #    'event.metadata.addressPostal',      ## Convert to uppercase, remove space.  # MAKE NOMINAL -> USE COUNT()?.
#         'event.metadata.addressProvince',   ## Convert provinces and states to code.  # CATEGORICAL.
#     #    'event.metadata.addressStreet',     ## Remove leading/trailing spaces. COUNT() may give information.  # CATEGORICAL.
#     #    'event.metadata.against', # Only USD.
#     #    'event.metadata.alphaPointUserId',
#         'event.metadata.amount', # FLOAT
#         'event.metadata.apTransferResponse.errorcode', # CATEGORICAL.
#         'event.metadata.apTransferResponse.errormsg',  # CATEGORICAL.
#         'event.metadata.apTransferResponse.result', # CATEGORICAL.
#     #    'event.metadata.authResponseAP.AddtlInfo',
#         'event.metadata.authResponseAP.AuthType',  # CATEGORICAL.
#         'event.metadata.authResponseAP.Authenticated', # CATEGORICAL.
#     #    'event.metadata.authResponseAP.Requires2FA',
#     #    'event.metadata.authResponseAP.SessionToken', # CATEGORICAL.
#     #    'event.metadata.authResponseAP.UserId', # CATEGORICAL.
#     #    'event.metadata.authResponseAP.errormsg',
#     #    'event.metadata.authResponseAP.twoFaToken',
#     #    'event.metadata.authResponseEIN.body',
#     #    'event.metadata.authResponseEIN.body.data.access_token',
#     #    'event.metadata.authResponseEIN.body.data.expires_in',
#     #    'event.metadata.authResponseEIN.body.data.refresh_token',
#     #   'event.metadata.authResponseEIN.body.data.scope',
#     #    'event.metadata.authResponseEIN.body.data.token_type',
#     #    'event.metadata.authResponseEIN.body.error',
#         'event.metadata.authResponseEIN.body.message',   ## Keep only first N characters.  # CATEGORICAL.
#     #    'event.metadata.authResponseEIN.body.status',
#     #    'event.metadata.authResponseEIN.body.success',
#         'event.metadata.authResponseEIN.bodyText',  ## Need to extract reponse categories.
#     #    'event.metadata.authResponseEIN.headers.map.',
#     #    'event.metadata.authResponseEIN.headers.map.Access-Control-Allow-Origin',
#     #    'event.metadata.authResponseEIN.headers.map.CF-RAY',
#     #    'event.metadata.authResponseEIN.headers.map.Connection',
#         'event.metadata.authResponseEIN.headers.map.Content-Length',  # CATEGORICAL.
#     #    'event.metadata.authResponseEIN.headers.map.Content-Type',
#     #    'event.metadata.authResponseEIN.headers.map.Date', ## To DATETIME.
#     #    'event.metadata.authResponseEIN.headers.map.ETag',
#     #    'event.metadata.authResponseEIN.headers.map.Etag',
#     #    'event.metadata.authResponseEIN.headers.map.Server',
#     #    'event.metadata.authResponseEIN.headers.map.X-Firefox-Spdy',
#     #    'event.metadata.authResponseEIN.headers.map.X-Powered-By',
#     #    'event.metadata.authResponseEIN.headers.map.access-control-allow-origin',
#         'event.metadata.authResponseEIN.headers.map.age',   ##  # CATEGORICAL.
#         'event.metadata.authResponseEIN.headers.map.cf-ray', ## ASK.
#     #    'event.metadata.authResponseEIN.headers.map.connection',
#     #    'event.metadata.authResponseEIN.headers.map.content-encoding',
#         'event.metadata.authResponseEIN.headers.map.content-length', # CATEGORICAL.
#     #    'event.metadata.authResponseEIN.headers.map.content-type',
#     #    'event.metadata.authResponseEIN.headers.map.date', ## To DATETIME.
#     #    'event.metadata.authResponseEIN.headers.map.etag',   # CATEGORICAL.
#     #    'event.metadata.authResponseEIN.headers.map.expect-ct',
#     #    'event.metadata.authResponseEIN.headers.map.server',
#         'event.metadata.authResponseEIN.headers.map.status',  # CATEGORICAL.
#     #    'event.metadata.authResponseEIN.headers.map.x-powered-by',
#         'event.metadata.authResponseEIN.ok',  # CATEGORICAL.
#         'event.metadata.authResponseEIN.status', # CATEGORICAL.
#         'event.metadata.authResponseEIN.statusText', # CATEGORICAL.
#     #    'event.metadata.authResponseEIN.url',
#     #    'event.metadata.bankAccountNumber',
#     #    'event.metadata.bankAddress',
#     #    'event.metadata.bankName',
#         'event.metadata.blockioResponse.data.amount_sent',
#         'event.metadata.blockioResponse.data.amount_withdrawn',
#     #    'event.metadata.blockioResponse.data.available_balance',
#     #    'event.metadata.blockioResponse.data.blockio_fee',
#         'event.metadata.blockioResponse.data.error_message', # Keep first N characters.
#         'event.metadata.blockioResponse.data.estimated_network_fee',
#     #    'event.metadata.blockioResponse.data.max_withdrawal_available',
#     #    'event.metadata.blockioResponse.data.minimum_balance_needed',
#         'event.metadata.blockioResponse.data.network',
#         'event.metadata.blockioResponse.data.network_fee',
#     #    'event.metadata.blockioResponse.data.txid',
#         'event.metadata.blockioResponse.status',  # CATEGORICAL.
#     #    'event.metadata.cardHolder',
#     #    'event.metadata.cardId',
#     #    'event.metadata.cardName',
#     #    'event.metadata.cardNumberLastFour',
#     #    'event.metadata.cardPrefix',
#     #    'event.metadata.cardSuffix',
#         'event.metadata.cashAdvanceReimbursement',
#         'event.metadata.cents',
#         'event.metadata.city',  # CATEGORICAL. To lower case.
#     #    'event.metadata.comment',
#     #    'event.metadata.country',
#     #    'event.metadata.currency',
#     #    'event.metadata.customerNumber',
#     #    'event.metadata.einTransactionId',
#         'event.metadata.email',
#     #    'event.metadata.failureUrl',
#         'event.metadata.firstAmount',
#     #    'event.metadata.firstName',
#     #    'event.metadata.fraudulent', # FIND OUT WHAT THIS MEANS.
#     #    'event.metadata.fullName',
#         'event.metadata.instrument', # CATEGORICAL.
#     #    'event.metadata.invoiceNumber',
#     #    'event.metadata.ip', # CATEGORICAL.
#     #    'event.metadata.landingPageId',
#     #    'event.metadata.languageCode',
#     #   'event.metadata.lastName',
#         'event.metadata.lastTradedPx',
#     #    'event.metadata.mongoResponse.__v',
#     #    'event.metadata.mongoResponse._id',
#         'event.metadata.mongoResponse.amount',
#     #    'event.metadata.mongoResponse.einTransactionId',
#     #    'event.metadata.mongoResponse.email',
#         'event.metadata.mongoResponse.price',
#         'event.metadata.mongoResponse.product', # CATEGORICAL.
#     #    'event.metadata.mongoResponse.status',
#     #    'event.metadata.mongoResponse.transactionId',
#     #    'event.metadata.name',
#         'event.metadata.passwordLength',
#         'event.metadata.paymentMethod', # CATEGORICAL.
#     #    'event.metadata.paymentToken',
#     #    'event.metadata.phoneNumber',
#     #    'event.metadata.postal', # To upper case, remove spaces.
#         'event.metadata.price',
#     #    'event.metadata.processorResponse',
#         'event.metadata.product', # CATEGORICAL.
#         'event.metadata.productId', # CATEGORICAL.
#     #    'event.metadata.profileId',
# # PROCESSOR ERROR CODES ARE LEGACY AND WILL NOT BE USE.
#     #    'event.metadata.prossessorError.authCode',
#     #    'event.metadata.prossessorError.avsResponse',
#     #    'event.metadata.prossessorError.billingDetails.city',
#     #    'event.metadata.prossessorError.billingDetails.country',
#     #    'event.metadata.prossessorError.billingDetails.state',
#     #    'event.metadata.prossessorError.billingDetails.street',
#     #    'event.metadata.prossessorError.billingDetails.zip',
#     #    'event.metadata.prossessorError.card.cardExpiry.month',     # To DATETIME MONTH.
#     #    'event.metadata.prossessorError.card.cardExpiry.year',      # To DATETIME YEAR.
#     #    'event.metadata.prossessorError.card.lastDigits',
#     #    'event.metadata.prossessorError.card.type', # CATEGORICAL.
#     #    'event.metadata.prossessorError.code', # CATEGORICAL.
#     #    'event.metadata.prossessorError.currencyCode', # CATEGORICAL.
#     #    'event.metadata.prossessorError.cvvVerification', # CATEGORICAL.
#     #    'event.metadata.prossessorError.error.message', # CATEGORICAL. Keep first N characters.
#     #    'event.metadata.prossessorError.id',
#     #    'event.metadata.prossessorError.links',
#     #    'event.metadata.prossessorError.merchantDescriptor.dynamicDescriptor',
#     #    'event.metadata.prossessorError.merchantDescriptor.phone',
#     #    'event.metadata.prossessorError.merchantRefNum', # CATEGORICAL.
#     #    'event.metadata.prossessorError.message',  # CATEGORICAL. - Check the possible categories.
#     #    'event.metadata.prossessorError.riskReasonCode', # CATEGORICAL.
#     #    'event.metadata.prossessorError.status', # CATEGORICAL.
#     #    'event.metadata.prossessorError.txnTime', # To DATETIME.
# #
#         'event.metadata.prossessorResponse.acsURL', # CATEGORICAL. Check -> Strip first part to get main URL.
#         'event.metadata.prossessorResponse.acs_url',    # CATEGORICAL. Check -> Strip first part to get main URL.
#         'event.metadata.prossessorResponse.amount',
#     #    'event.metadata.prossessorResponse.authCode',
#     #    'event.metadata.prossessorResponse.authentication.cavv',
#         'event.metadata.prossessorResponse.authentication.eci',    # CATEGORICAL.
#         'event.metadata.prossessorResponse.authentication.signatureStatus',    # CATEGORICAL.
#         'event.metadata.prossessorResponse.authentication.threeDEnrollment',    # CATEGORICAL.
#         'event.metadata.prossessorResponse.authentication.threeDResult',    # CATEGORICAL.
#     #    'event.metadata.prossessorResponse.authentication.xid',
#         'event.metadata.prossessorResponse.avsResponse',    # CATEGORICAL.
#         'event.metadata.prossessorResponse.billingDetails.city',    # CATEGORICAL. To lower case.
#         'event.metadata.prossessorResponse.billingDetails.country',    # CATEGORICAL. Convert to code. Just CA/US right now.
#         'event.metadata.prossessorResponse.billingDetails.province',    # CATEGORICAL. Convert to province/state code.
#         'event.metadata.prossessorResponse.billingDetails.state',    # CATEGORICAL. Convert to province/state code.
#     #    'event.metadata.prossessorResponse.billingDetails.street',
#         'event.metadata.prossessorResponse.billingDetails.zip',    # CATEGORICAL. To uppercase. Remove spaces.
#         'event.metadata.prossessorResponse.card.cardExpiry.month',  # To DATETIME MONTH.
#         'event.metadata.prossessorResponse.card.cardExpiry.year',   # To DATETIME YEAR.
#         'event.metadata.prossessorResponse.card.cardType',    # CATEGORICAL.
#     #    'event.metadata.prossessorResponse.card.lastDigits',
#         'event.metadata.prossessorResponse.card.type',    # CATEGORICAL. Check conversion to card code.
#         'event.metadata.prossessorResponse.card_expiry_month',  # To DATETIME MONTH.
#         'event.metadata.prossessorResponse.card_expiry_year',   # To DATETIME YEAR.
#     #    'event.metadata.prossessorResponse.card_suffix',
#         'event.metadata.prossessorResponse.card_type',   # To DATETIME YEAR.
#     #    'event.metadata.prossessorResponse.cavv',
#         'event.metadata.prossessorResponse.charge_amount',
#     #    'event.metadata.prossessorResponse.code',
#     #    'event.metadata.prossessorResponse.currency',
#     #    'event.metadata.prossessorResponse.currencyCode',
#     #    'event.metadata.prossessorResponse.customerIp', ## NOT USED CURRENTLY. CONVERT TO LAT/LONG LATER.
#         'event.metadata.prossessorResponse.cvvVerification',    ## CATEGORICAL.
#     #    'event.metadata.prossessorResponse.description',
#     #    'event.metadata.prossessorResponse.eci',
#         'event.metadata.prossessorResponse.error.code',    ## CATEGORICAL.
#     #    'event.metadata.prossessorResponse.error.details',
#         'event.metadata.prossessorResponse.error.fieldErrors',    ## CATEGORICAL.
#         'event.metadata.prossessorResponse.error.links',    ## CATEGORICAL.
#         'event.metadata.prossessorResponse.error.message',    ## CATEGORICAL.
#     #    'event.metadata.prossessorResponse.externalId',
#     #    'event.metadata.prossessorResponse.id',
#         'event.metadata.prossessorResponse.links',    ## CATEGORICAL. LOOK THROUGH THE RESPONSES CAREFULLY. LUMP?
#         'event.metadata.prossessorResponse.merchantDescriptor.dynamicDescriptor',    ## CATEGORICAL.
#     #    'event.metadata.prossessorResponse.merchantDescriptor.phone',
#     #    'event.metadata.prossessorResponse.merchantRefNum',
#     #    'event.metadata.prossessorResponse.message',
#     #    'event.metadata.prossessorResponse.paReq',
#     #    'event.metadata.prossessorResponse.pa_req',
#     #    'event.metadata.prossessorResponse.profile.email',
#     #    'event.metadata.prossessorResponse.profile.firstName',
#     #    'event.metadata.prossessorResponse.profile.lastName',
#     #    'event.metadata.prossessorResponse.reference_number',
#     #    'event.metadata.prossessorResponse.request_ip',
#         'event.metadata.prossessorResponse.riskReasonCode',    ## CATEGORICAL.
#     #    'event.metadata.prossessorResponse.settleWithAuth',
#     #    'event.metadata.prossessorResponse.settlements',
#         'event.metadata.prossessorResponse.signatureStatus',    ## CATEGORICAL.
#         'event.metadata.prossessorResponse.status',    ## CATEGORICAL.
#         'event.metadata.prossessorResponse.threeDEnrollment',    ## CATEGORICAL.
#         'event.metadata.prossessorResponse.threeDResult',    ## CATEGORICAL.
#     #    'event.metadata.prossessorResponse.txnTime',    # To DATETIME.
#     #    'event.metadata.prossessorResponse.xid',
#         'event.metadata.province',    ## CATEGORICAL. Convert to province/state codes.
#         'event.metadata.rate',  # How to tie to the Coin type?
#         'event.metadata.requestParams.amount',
#     #    'event.metadata.requestParams.card_id',
#         'event.metadata.requestParams.charge_amount',
#     #    'event.metadata.requestParams.currency',
#     #    'event.metadata.requestParams.email',
#     #    'event.metadata.requestParams.enrollmentId',
#     #    'event.metadata.requestParams.enrollment_id',
#         'event.metadata.requestParams.enrollment_status',    ## CATEGORICAL.
#     #    'event.metadata.requestParams.paymentToken',
#         'event.metadata.requestParams.price',
#         'event.metadata.requestParams.product',    ## CATEGORICAL.
#         'event.metadata.requestParams.product_amount',
#     #    'event.metadata.requestParams.profile_id',
#         'event.metadata.requestParams.redirectUrl',    ## CATEGORICAL.
#         'event.metadata.requestParams.redirect_url',    ## CATEGORICAL.
#     #    'event.metadata.requestParams.reference_number',
#         'event.metadata.requestParams.threeDEnrollment',    ## CATEGORICAL.
#         'event.metadata.secondAmount',
#     #    'event.metadata.secureId',
#     #    'event.metadata.sessionId',
#         'event.metadata.side',    ## CATEGORICAL.
#         'event.metadata.status',    ## CATEGORICAL.
#     #    'event.metadata.street',
#         'event.metadata.successUrl',    ## CATEGORICAL.
#         'event.metadata.tradesResponse',    ## CATEGORICAL.
#     #    'event.metadata.transactionId',
#     #    'event.metadata.transaction_id',
#         'event.metadata.twoFactorResponse.Authenticated',    ## CATEGORICAL.
#     #    'event.metadata.twoFactorResponse.SessionToken',
#     #    'event.metadata.twoFactorResponse.UserId',
#         'event.metadata.type',    ## CATEGORICAL.
#     #    'event.metadata.updated',
#         'event.metadata.userAgent',    ## CATEGORICAL. Truncate after first N characters?
#     #    'event.metadata.vaultId',
#     #    'event.metadata.wallet',    ## CATEGORICAL.
#         'event.value',
#     #    'request.__v',
#         'request._id',
#         'request.created',  # To DATETIME.
#         'request.eventAction',    ## CATEGORICAL.
#         'request.eventCategory',    ## CATEGORICAL.
#         'request.eventLabel',    ## CATEGORICAL.
#         'request.metadata.addressCity',    ## CATEGORICAL.
#         'request.metadata.addressCountry',    ## CATEGORICAL.
#     #    'request.metadata.addressPostal',    ## CATEGORICAL.
#         'request.metadata.addressProvince',    ## CATEGORICAL.
#     #    'request.metadata.addressStreet',
#         'request.metadata.amount',
#     #    'request.metadata.apTransferResponse.errorcode',
#     #    'request.metadata.apTransferResponse.result',
#     #    'request.metadata.cardName',
#     #    'request.metadata.cardNumberLastFour',
#         'request.metadata.cashAdvanceReimbursement',
#         'request.metadata.cents',
#     #    'request.metadata.customerNumber',
#     #    'request.metadata.einTransactionId',
#         'request.metadata.email', # INDEX VARIABLE.
#         'request.metadata.failureUrl',    ## CATEGORICAL.
#     #    'request.metadata.firstName',
#     #    'request.metadata.fraudulent', # *** HOW TO USE THIS? *** DON'T.
#     #    'request.metadata.invoiceNumber',
#     #    'request.metadata.ip',
#     #    'request.metadata.languageCode',
#     #    'request.metadata.lastName',
#     #    'request.metadata.mongoResponse.__v',
#     #    'request.metadata.mongoResponse._id',
#         'request.metadata.mongoResponse.amount',
#     #    'request.metadata.mongoResponse.einTransactionId',
#     #    'request.metadata.mongoResponse.email',
#         'request.metadata.mongoResponse.price',
#         'request.metadata.mongoResponse.product',    ## CATEGORICAL.
#         'request.metadata.mongoResponse.status',    ## CATEGORICAL.
#     #    'request.metadata.mongoResponse.transactionId',
#     #    'request.metadata.paymentToken',
#         'request.metadata.price',
#         'request.metadata.processorResponse',    ## CATEGORICAL. Truncate after first N characters.
#         'request.metadata.product',    ## CATEGORICAL.
#         'request.metadata.prossessorResponse.avsResponse',    ## CATEGORICAL.
#         'request.metadata.prossessorResponse.billingDetails.city',    ## CATEGORICAL. Lower case, remove lead/trailing spaces.
#         'request.metadata.prossessorResponse.billingDetails.country',    ## CATEGORICAL. CODE.
#         'request.metadata.prossessorResponse.billingDetails.province',    ## CATEGORICAL. CODE.
#     #    'request.metadata.prossessorResponse.billingDetails.street',
#     #    'request.metadata.prossessorResponse.billingDetails.zip',    ## CATEGORICAL. CODE.
#         'request.metadata.prossessorResponse.card.cardExpiry.month', # TIMEDATE MONTH.
#         'request.metadata.prossessorResponse.card.cardExpiry.year', # TIMEDATE YEAR.
#     #    'request.metadata.prossessorResponse.card.lastDigits',
#         'request.metadata.prossessorResponse.card.type',    ## CATEGORICAL.
#     #    'request.metadata.prossessorResponse.currencyCode',
#     #    'request.metadata.prossessorResponse.externalId',
#     #    'request.metadata.prossessorResponse.merchantRefNum',
#         'request.metadata.prossessorResponse.status',    ## CATEGORICAL.
#         'request.metadata.rate',
#         'request.metadata.requestParams.amount',
#     #    'request.metadata.requestParams.card_id',
#         'request.metadata.requestParams.charge_amount',
#     #    'request.metadata.requestParams.currency',
#     #    'request.metadata.requestParams.email',
#     #    'request.metadata.requestParams.enrollmentId',
#     #    'request.metadata.requestParams.enrollment_id',
#         'request.metadata.requestParams.enrollment_status',    ## CATEGORICAL.
#     #    'request.metadata.requestParams.paymentToken',
#         'request.metadata.requestParams.price',
#         'request.metadata.requestParams.product',    ## CATEGORICAL.
#         'request.metadata.requestParams.product_amount',
#     #    'request.metadata.requestParams.profile_id',
#         'request.metadata.requestParams.redirectUrl',    ## CATEGORICAL.
#         'request.metadata.requestParams.redirect_url',    ## CATEGORICAL.
#     #    'request.metadata.requestParams.reference_number',
#         'request.metadata.requestParams.threeDEnrollment',    ## CATEGORICAL.
#     #    'request.metadata.secureId',
#         'request.metadata.status',    ## CATEGORICAL.
#         'request.metadata.successUrl',    ## CATEGORICAL.
#     #    'request.metadata.transactionId',
#     #    'request.metadata.transaction_id',
#         'request.metadata.type',    ## CATEGORICAL.
#     #    'request.metadata.updated', # Check if useful.
#         'request.metadata.userAgent',    ## CATEGORICAL. Keep first N characters.
#     #    'request.metadata.wallet',
#         'request.value'
#     ]

# #    df = df.drop(labels=columns_to_keep_list, axis=1, errors='ignore')
#     df = df.loc[ :, columns_to_keep_list ]
        
#     return df