problemas = [#"25FV47", # Ponto inicial não é viavel
 #            "ADLITTLE" , # Ponto inicial não é viavel
             "AFIRO" , # viavel
 #            "AGG"    ,  # Ponto inicial não é viavel
 #            "AGG2"    , # Ponto inicial não é viavel
 #            "AGG3"     , # Ponto inicial não é viavel
 #            "BANDM"    , # Ponto inicial não é viavel
 #            "BEACONFD" , # Ponto inicial não é viavel
             ##    "BLEND"    , # esse problema não está lendo corretamente
#             "BNL1"     , # Ponto inicial não é viavel
#             "BNL2"     , # Ponto inicial não é viavel
##             "BRANDY"   , # está com problema no carregamento também
#             "D2Q06C"   , # Ponto inicial não é viavel
#             "DEGEN2"   , # Ponto inicial não é viavel
#             "DEGEN3"   , # Ponto inicial não é viavel
#             "E226"     , # Ponto inicial não é viavel
#             "FFFFF800" , # Ponto inicial não é viavel (percebi o problema com o ponto inicial testando esse)
#             "ISRAEL"   , # Ponto inicial não é viavel
#             "LOTFI"    , # Ponto inicial não é viavel
             "MAROS-R7" , # viavel
             "QAP8"     , # ponto viável, mas ocorre singular exception
             "QAP12"    , # viavel (embora dê singular exception)
             ##    "QAP15"    , # esse problema é pesado demais e provoca a morte do processo por falta de memoria
#             "SC105"    , # Ponto inicial não é viavel
#             "SC205"    , # Ponto inicial não é viavel
#             "SC50A"    ,# Ponto inicial não é viavel
#             "SC50B"    , # Ponto inicial não é viavel
#             "SCAGR25"  , # Ponto inicial não é viavel
             "SCAGR7"   , # viavel
#             "SCFXM1"   , # Ponto inicial não é viavel
#             "SCFXM2"   , # Ponto inicial não é viavel
#             "SCFXM3"   , # Ponto inicial não é viavel
#             "SCORPION" , # Ponto inicial não é viavel
#             "SCRS8"    , # Ponto inicial não é viavel
             "SCSD1"    , # viavel
             "SCSD6"    , # viavel
             "SCSD8"    , # viavel
             "SCTAP1"   , # viavel
             "SCTAP2"   , # viavel
             "SCTAP3"   , # viavel
#             "SHARE1B"  ,# Ponto inicial não é viavel
             "SHARE2B"  , # viavel
#             "SHIP04L"  ,# Ponto inicial não é viavel
#             "SHIP04S"  ,# Ponto inicial não é viavel
#             "SHIP08L"  ,# Ponto inicial não é viavel
#             "SHIP08S"  ,# Ponto inicial não é viavel
#             "SHIP12L"  ,# Ponto inicial não é viavel
#             "SHIP12S"  , # Ponto inicial não é viavel
             "STOCFOR1" , # viavel
             "STOCFOR2" , # viavel
#             "STOCFOR3" , # out of memory
             "TRUSS"    ]#, # Viavel mas pesado, pode ser que de certo...
#             "WOOD1P"   , # Ponto inicial não é viavel
#             "WOODW" ] # Ponto inicial não é viavel

println("Primeiramente, inclua algum dos arquivos testar_{implementacao}.jl")

println("Depois, rode test_all()")

wait_for_key(prompt) = (print(stdout, prompt); read(stdin, 1); nothing)

function test_all(ignore = false, pausas = false, skiptiny = false, skipto = 1)

    if skiptiny == false
        for i=1:3
            println("")
            println(" >>> Testando o problema $(i).")
            @time begin
                testar(i, ignore)
            end
            println("")
            if pausas == true
                wait_for_key("Tecle ENTER para continuar...")
            end
        end
        skipto = 1
    end
    for i = skipto:length(problemas)
        println("")
        println(" >>> Testando o problema $(problemas[i]).")
        @time begin
            testar(problemas[i], ignore)
        end
        println("")
        if pausas == true
            wait_for_key("Tecle ENTER para continuar...")
        end
    end
end
